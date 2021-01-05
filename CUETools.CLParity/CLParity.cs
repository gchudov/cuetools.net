/**
 * CUETools.CLParity: Reed-Solomon (32 bit) using OpenCL
 * Copyright (c) 2009-2021 Gregory S. Chudov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Text;
using System.Runtime.InteropServices;
using CUETools.Codecs;
using OpenCLNet;

namespace CUETools.CLParity
{
    public class CLParitySettings
    {
		public CLParitySettings() 
        { 
            this.MappedMemory = false;
            this.GroupSize = 128;
            this.DeviceType = OpenCLDeviceType.GPU;
        }

        [DefaultValue(false)]
        [SRDescription(typeof(Properties.Resources), "DescriptionMappedMemory")]
        public bool MappedMemory { get; set; }

		[TypeConverter(typeof(CLParitySettingsGroupSizeConverter))]
        [DefaultValue(128)]
        [SRDescription(typeof(Properties.Resources), "DescriptionGroupSize")]
        public int GroupSize { get; set; }

        [SRDescription(typeof(Properties.Resources), "DescriptionDefines")]
        public string Defines { get; set; }

		[TypeConverter(typeof(CLParitySettingsPlatformConverter))]
        [SRDescription(typeof(Properties.Resources), "DescriptionPlatform")]
        public string Platform { get; set; }

        [DefaultValue(OpenCLDeviceType.GPU)]
        [SRDescription(typeof(Properties.Resources), "DescriptionDeviceType")]
        public OpenCLDeviceType DeviceType { get; set; }
    }

	public class CLParitySettingsPlatformConverter : TypeConverter
    {
        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {            
            var res = new List<string>();
            foreach (var p in OpenCL.GetPlatforms())
                res.Add(p.Name);
            return new StandardValuesCollection(res);
        }
    }

	public class CLParitySettingsGroupSizeConverter : TypeConverter
    {
        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            return new StandardValuesCollection(new int[] { 64, 128, 256 });
        }
    }

    public enum OpenCLDeviceType : ulong
    {
        CPU = DeviceType.CPU,
        GPU = DeviceType.GPU
    }

	//[AudioEncoderClass("CLParity", typeof(CLParitySettings))]
    public class CLParityWriter : IAudioDest
    {
        long _position;

        // total stream samples
        // if 0, stream length is unknown
        int sample_count = -1;

        TimeSpan _userProcessorTime;

        int samplesInBuffer = 0;

        bool inited = false;

        OpenCLManager OCLMan;
        Program openCLProgram;

        CLParityTask task1;
		CLParityTask task2;

		int npar, stride, stridesPerTask;

        AudioPCMConfig _pcm;

        public CLParityWriter(string path, Stream IO, AudioPCMConfig pcm)
        {
            _pcm = pcm;
            if (pcm.BitsPerSample != 16)
                throw new Exception("Bits per sample must be 16.");
            if (pcm.ChannelCount != 2)
                throw new Exception("ChannelCount must be 2.");
			npar = 256;
			stride = 1;
        }

		public CLParityWriter(string path, AudioPCMConfig pcm)
            : this(path, null, pcm)
        {
        }

		internal CLParitySettings _settings = new CLParitySettings();

        public object Settings
        {
            get
            {
                return _settings;
            }
            set
            {
				if (value as CLParitySettings == null)
                    throw new Exception("Unsupported options " + value);
				_settings = value as CLParitySettings;
            }
        }

        //[DllImport("kernel32.dll")]
        //static extern bool GetThreadTimes(IntPtr hThread, out long lpCreationTime, out long lpExitTime, out long lpKernelTime, out long lpUserTime);
        //[DllImport("kernel32.dll")]
        //static extern IntPtr GetCurrentThread();

        void DoClose()
        {
            if (inited)
            {
                int strideCount = samplesInBuffer / stride;
				if (strideCount > 0)
					do_output_frames(strideCount);
                if (samplesInBuffer > 0)
					throw new Exception(string.Format("samplesInBuffer % stride != 0"));
                if (task2.strideCount > 0)
                {
                    task2.openCLCQ.Finish(); // cuda.SynchronizeStream(task2.stream);
					task2.strideCount = 0;
                }

                task1.Dispose();
                task2.Dispose();
                openCLProgram.Dispose();
                OCLMan.Dispose();
                inited = false;
            }
        }

        public void Close()
        {
            DoClose();
            if (sample_count > 0 && _position != sample_count)
                throw new Exception(string.Format("Samples written differs from the expected sample count. Expected {0}, got {1}.", sample_count, _position));
        }

        public void Delete()
        {
            if (inited)
            {
                task1.Dispose();
                task2.Dispose();
                openCLProgram.Dispose();
                OCLMan.Dispose();
                inited = false;
            }
        }

        public long Position
        {
            get
            {
                return _position;
            }
        }

        public long FinalSampleCount
        {
            set { sample_count = (int)value; }
        }

        public TimeSpan UserProcessorTime
        {
            get { return _userProcessorTime; }
        }

        public AudioPCMConfig PCM
        {
            get { return _pcm; }
        }

        public unsafe void InitTasks()
        {
            if (!inited)
            {
                if (OpenCL.NumberOfPlatforms < 1)
                    throw new Exception("no opencl platforms found");

                int groupSize = _settings.DeviceType == OpenCLDeviceType.CPU ? 1 : _settings.GroupSize;
                OCLMan = new OpenCLManager();
                // Attempt to save binaries after compilation, as well as load precompiled binaries
                // to avoid compilation. Usually you'll want this to be true. 
                OCLMan.AttemptUseBinaries = true; // true;
                // Attempt to compile sources. This should probably be true for almost all projects.
                // Setting it to false means that when you attempt to compile "mysource.cl", it will
                // only scan the precompiled binary directory for a binary corresponding to a source
                // with that name. There's a further restriction that the compiled binary also has to
                // use the same Defines and BuildOptions
                OCLMan.AttemptUseSource = true;
                // Binary and source paths
                // This is where we store our sources and where compiled binaries are placed
                //OCLMan.BinaryPath = @"OpenCL\bin";
                //OCLMan.SourcePath = @"OpenCL\src";
                // If true, RequireImageSupport will filter out any devices without image support
                // In this project we don't need image support though, so we set it to false
                OCLMan.RequireImageSupport = false;
                // The BuildOptions string is passed directly to clBuild and can be used to do debug builds etc
                OCLMan.BuildOptions = "";
                OCLMan.SourcePath = System.IO.Path.GetDirectoryName(GetType().Assembly.Location);
                OCLMan.BinaryPath = System.IO.Path.Combine(System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "CUE Tools"), "OpenCL");
                int platformId = 0;
                if (_settings.Platform != null)
                {
                    platformId = -1;
                    string platforms = "";
                    for (int i = 0; i < OpenCL.NumberOfPlatforms; i++)
                    {
                        var platform = OpenCL.GetPlatform(i);
                        platforms += " \"" + platform.Name + "\"";
                        if (platform.Name.Equals(_settings.Platform, StringComparison.InvariantCultureIgnoreCase))
                        {
                            platformId = i;
                            break;
                        }
                    }
                    if (platformId < 0)
                        throw new Exception("unknown platform \"" + _settings.Platform + "\". Platforms available:" + platforms);
                }
                OCLMan.CreateDefaultContext(platformId, (DeviceType)_settings.DeviceType);

				this.stridesPerTask = (int)OCLMan.Context.Devices[0].MaxComputeUnits * npar * 8;

                // The Defines string gets prepended to any and all sources that are compiled
                // and serve as a convenient way to pass configuration information to the compilation process
                OCLMan.Defines =
                    "#define GROUP_SIZE " + groupSize.ToString() + "\n" +
                    "#define CLPARITY_VERSION \"" + vendor_string + "\"\n" +
#if DEBUG
                    "#define DEBUG\n" +
#endif
                    (_settings.DeviceType == OpenCLDeviceType.CPU ? "#define CLPARITY_CPU\n" : "") +
                    _settings.Defines + "\n";

                var exts = new string[] { "cl_khr_local_int32_base_atomics", "cl_khr_local_int32_extended_atomics", "cl_khr_fp64", "cl_amd_fp64" };
                foreach (string extension in exts)
                    if (OCLMan.Context.Devices[0].Extensions.Contains(extension))
                    {
                        OCLMan.Defines += "#pragma OPENCL EXTENSION " + extension + ": enable\n";
                        OCLMan.Defines += "#define HAVE_" + extension + "\n";
                    }

                try
                {
                    openCLProgram = OCLMan.CompileFile("parity.cl");
                }
                catch (OpenCLBuildException ex)
                {
                    string buildLog = ex.BuildLogs[0];
                    throw ex;
                }
                //using (Stream kernel = GetType().Assembly.GetManifestResourceStream(GetType(), "parity.cl"))
                //using (StreamReader sr = new StreamReader(kernel))
                //{
                //    try
                //    {
                //        openCLProgram = OCLMan.CompileSource(sr.ReadToEnd()); ;
                //    }
                //    catch (OpenCLBuildException ex)
                //    {
                //        string buildLog = ex.BuildLogs[0];
                //        throw ex;
                //    }
                //}
#if TTTTKJHSKJH
                var openCLPlatform = OpenCL.GetPlatform(0);
                openCLContext = openCLPlatform.CreateDefaultContext();
                using (Stream kernel = GetType().Assembly.GetManifestResourceStream(GetType(), "parity.cl"))
                using (StreamReader sr = new StreamReader(kernel))
                    openCLProgram = openCLContext.CreateProgramWithSource(sr.ReadToEnd());
                try
                {
                    openCLProgram.Build();
                }
                catch (OpenCLException)
                {
                    string buildLog = openCLProgram.GetBuildLog(openCLProgram.Devices[0]);
                    throw;
                }
#endif

				task1 = new CLParityTask(openCLProgram, this, groupSize, this.npar, this.stride, this.stridesPerTask);
				task2 = new CLParityTask(openCLProgram, this, groupSize, this.npar, this.stride, this.stridesPerTask);
                inited = true;
            }
        }

        public unsafe void Write(AudioBuffer buff)
        {
            InitTasks();
            buff.Prepare(this);
            int pos = 0;
            while (pos < buff.Length)
            {
				int block = Math.Min(buff.Length - pos, stride * stridesPerTask - samplesInBuffer);

                fixed (byte* buf = buff.Bytes)
                    AudioSamples.MemCpy(((byte*)task1.clSamplesBytesPtr) + samplesInBuffer * _pcm.BlockAlign, buf + pos * _pcm.BlockAlign, block * _pcm.BlockAlign);
                
                samplesInBuffer += block;
                pos += block;

                int strideCount = samplesInBuffer / stride;
				if (strideCount >= stridesPerTask)
					do_output_frames(strideCount);
            }
        }

		public unsafe void do_output_frames(int strideCount)
        {
			task1.strideCount = strideCount;
			if (!task1.UseMappedMemory)
				task1.openCLCQ.EnqueueWriteBuffer(task1.clSamplesBytes, false, 0, sizeof(int) * stride * strideCount, task1.clSamplesBytesPtr);
			//task.openCLCQ.EnqueueUnmapMemObject(task.clSamplesBytes, task.clSamplesBytes.HostPtr);
			//task.openCLCQ.EnqueueMapBuffer(task.clSamplesBytes, true, MapFlags.WRITE, 0, task.samplesBufferLen / 2);
			task1.EnqueueKernels();
			if (task2.strideCount > 0)
                task2.openCLCQ.Finish();
			int bs = stride * strideCount;
            samplesInBuffer -= bs;
            if (samplesInBuffer > 0)
                AudioSamples.MemCpy(
                    ((byte*)task2.clSamplesBytesPtr),
                    ((byte*)task1.clSamplesBytesPtr) + bs * _pcm.BlockAlign, 
                    samplesInBuffer * _pcm.BlockAlign);
			CLParityTask tmp = task1;
            task1 = task2;
            task2 = tmp;
			task1.strideCount = 0;
        }

        public string Path { get { return null; } }

        public static readonly string vendor_string = "CLParity#2.1.7";
    }

	internal class CLParityTask
	{
		Program openCLProgram;
		public CommandQueue openCLCQ;
		public Kernel reedSolomonInit;
		public Kernel reedSolomonInitGx;
		public Kernel reedSolomonA;
		public Kernel reedSolomonB;
		public Kernel reedSolomon;
		public Kernel reedSolomonDecodeInit;
		public Kernel reedSolomonDecode;
		public Kernel chienSearch;
		public Mem clSamplesBytes;
		public Mem clExp;
		public Mem clEncodeGx0;
		public Mem clEncodeGx1;
		public Mem clParity0;
		public Mem clParity1;
		public Mem clSigma;
		public Mem clWkOut;

		public Mem clSamplesBytesPinned;
		public Mem clExpPinned;
		public Mem clParity0Pinned;
		public Mem clParity1Pinned;

		public IntPtr clSamplesBytesPtr;
		public IntPtr clExpPtr;
		public IntPtr clParity0Ptr;
		public IntPtr clParity1Ptr;

		public int[] samplesBuffer;
		public int strideCount = 0;
		public int npar;
		public int stride;
		public int maxStridesCount;

		public Thread workThread = null;
		public Exception exception = null;
		public bool done = false;
		public bool exit = false;

		public int groupSize = 128;
		public CLParityWriter writer;
		public bool UseMappedMemory = false;

		unsafe public CLParityTask(Program _openCLProgram, CLParityWriter writer, int groupSize, int npar, int stride, int maxStridesCount)
		{
			this.UseMappedMemory = writer._settings.MappedMemory || writer._settings.DeviceType == OpenCLDeviceType.CPU;
			this.groupSize = groupSize;
			this.writer = writer;
			openCLProgram = _openCLProgram;
#if DEBUG
            var prop = CommandQueueProperties.PROFILING_ENABLE;
#else
			var prop = CommandQueueProperties.NONE;
#endif
			openCLCQ = openCLProgram.Context.CreateCommandQueue(openCLProgram.Context.Devices[0], prop);

			this.npar = npar;
			this.stride = stride;
			this.maxStridesCount = maxStridesCount;

			int samplesBufferLen = this.maxStridesCount * this.stride * sizeof(int);
			int parityLength = (this.npar + 1) * sizeof(int);
			int encodeGxLength = (this.npar + this.groupSize) * sizeof(int);
			int expLength = this.npar * sizeof(int);
			int sigmaLength = this.npar * sizeof(int);
			//int wkOutLength = this.npar * sizeof(int);

			if (!this.UseMappedMemory)
			{
				clSamplesBytes = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, samplesBufferLen);
				clParity0 = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, parityLength);
				clParity1 = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, parityLength);
				clExp = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, expLength);

				clSamplesBytesPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, samplesBufferLen);
				clParity0Pinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, parityLength);
				clParity1Pinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, parityLength);
				clExpPinned = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, expLength);

				clSamplesBytesPtr = openCLCQ.EnqueueMapBuffer(clSamplesBytesPinned, true, MapFlags.READ_WRITE, 0, samplesBufferLen);
				clParity0Ptr = openCLCQ.EnqueueMapBuffer(clParity0Pinned, true, MapFlags.READ_WRITE, 0, parityLength);
				clParity1Ptr = openCLCQ.EnqueueMapBuffer(clParity1Pinned, true, MapFlags.READ_WRITE, 0, parityLength);
				clExpPtr = openCLCQ.EnqueueMapBuffer(clExpPinned, true, MapFlags.READ_WRITE, 0, expLength);
			}
			else
			{
				clSamplesBytes = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, samplesBufferLen);
				clParity0 = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, parityLength);
				clParity1 = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, parityLength);
				clExp = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE | MemFlags.ALLOC_HOST_PTR, expLength);

				clSamplesBytesPtr = openCLCQ.EnqueueMapBuffer(clSamplesBytes, true, MapFlags.READ_WRITE, 0, samplesBufferLen);
				clParity0Ptr = openCLCQ.EnqueueMapBuffer(clParity0, true, MapFlags.READ_WRITE, 0, parityLength);
				clParity1Ptr = openCLCQ.EnqueueMapBuffer(clParity1, true, MapFlags.READ_WRITE, 0, parityLength);
				clExpPtr = openCLCQ.EnqueueMapBuffer(clExp, true, MapFlags.READ_WRITE, 0, expLength);

				//clSamplesBytesPtr = clSamplesBytes.HostPtr;
				//clResidualPtr = clResidual.HostPtr;
				//clBestRiceParamsPtr = clBestRiceParams.HostPtr;
				//clResidualTasksPtr = clResidualTasks.HostPtr;
				//clWindowFunctionsPtr = clWindowFunctions.HostPtr;
			}

			clEncodeGx0 = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, encodeGxLength);
			clEncodeGx1 = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, encodeGxLength);

			//clSamples = openCLProgram.Context.CreateBuffer(MemFlags.READ_WRITE, samplesBufferLen);
			//openCLCQ.EnqueueMapBuffer(clSamplesBytes, true, MapFlags.WRITE, 0, samplesBufferLen / 2);

			reedSolomonInit = openCLProgram.CreateKernel("reedSolomonInit");
			reedSolomonInitGx = openCLProgram.CreateKernel("reedSolomonInitGx");
			reedSolomonA = openCLProgram.CreateKernel("reedSolomonA");
			reedSolomonB = openCLProgram.CreateKernel("reedSolomonB");
			reedSolomon = openCLProgram.CreateKernel("reedSolomon");
			reedSolomonDecodeInit = openCLProgram.CreateKernel("reedSolomonDecodeInit");
			reedSolomonDecode = openCLProgram.CreateKernel("reedSolomonDecode");
			chienSearch = openCLProgram.CreateKernel("chienSearch");

			samplesBuffer = new int[this.maxStridesCount * this.stride];

			InitData();
		}

		public void Dispose()
		{
			if (workThread != null)
			{
				lock (this)
				{
					exit = true;
					Monitor.Pulse(this);
				}
				workThread.Join();
				workThread = null;
			}

			openCLCQ.Finish();

			reedSolomonInit.Dispose();
			reedSolomonInitGx.Dispose();
			reedSolomonA.Dispose();
			reedSolomonB.Dispose();
			reedSolomon.Dispose();
			reedSolomonDecodeInit.Dispose();
			reedSolomonDecode.Dispose();
			chienSearch.Dispose();

			if (!this.UseMappedMemory)
			{
				if (clSamplesBytesPtr != IntPtr.Zero)
					openCLCQ.EnqueueUnmapMemObject(clSamplesBytesPinned, clSamplesBytesPtr);
				clSamplesBytesPtr = IntPtr.Zero;
				if (clParity0Ptr != IntPtr.Zero)
					openCLCQ.EnqueueUnmapMemObject(clParity0Pinned, clParity0Ptr);
				clParity0Ptr = IntPtr.Zero;
				if (clParity1Ptr != IntPtr.Zero)
					openCLCQ.EnqueueUnmapMemObject(clParity1Pinned, clParity1Ptr);
				clParity1Ptr = IntPtr.Zero;
				if (clExpPtr != IntPtr.Zero)
					openCLCQ.EnqueueUnmapMemObject(clExpPinned, clExpPtr);
				clExpPtr = IntPtr.Zero;

				clSamplesBytesPinned.Dispose();
				clParity0Pinned.Dispose();
				clParity1Pinned.Dispose();
				clExpPinned.Dispose();
			}
			else
			{
				openCLCQ.EnqueueUnmapMemObject(clSamplesBytes, clSamplesBytesPtr);
				openCLCQ.EnqueueUnmapMemObject(clParity0, clParity0Ptr);
				openCLCQ.EnqueueUnmapMemObject(clParity1, clParity1Ptr);
				openCLCQ.EnqueueUnmapMemObject(clExp, clExpPtr);
			}

			clSamplesBytes.Dispose();
			clParity0.Dispose();
			clParity1.Dispose();
			clEncodeGx0.Dispose();
			clEncodeGx1.Dispose();
			clExp.Dispose();

			//clSamples.Dispose();

			openCLCQ.Dispose();

			GC.SuppressFinalize(this);
		}

		private unsafe void InitData()
		{
			reedSolomonInit.SetArgs(
				clEncodeGx0,
				clEncodeGx1,
				clParity0,
				clParity1,
				npar);

			openCLCQ.EnqueueNDRangeKernel(
				reedSolomonInit,
				groupSize, npar / groupSize);

			for (int i = 0; i < npar / (groupSize / 2); i++)
			{
				reedSolomonInitGx.SetArgs(
					clExp,
					clEncodeGx0,
					clEncodeGx1,
					npar,
					i);

				openCLCQ.EnqueueNDRangeKernel(
					reedSolomonInitGx,
					groupSize, npar / (groupSize / 2));

				var temp = clEncodeGx0; clEncodeGx0 = clEncodeGx1; clEncodeGx1 = temp;
			}
		}

		internal unsafe void EnqueueKernels()
		{
			int blocks = strideCount / groupSize;
			for (int i = 0; i < blocks; i++)
			{
				reedSolomonA.SetArgs(clSamplesBytes, clEncodeGx0, clParity0, this.npar, i * groupSize);
				openCLCQ.EnqueueNDRangeKernel(reedSolomonA, groupSize, 1);
				reedSolomonB.SetArgs(clSamplesBytes, clEncodeGx0, clParity0, this.npar, i * groupSize);
				openCLCQ.EnqueueNDRangeKernel(reedSolomonB, groupSize, npar / groupSize);
			}
			for (int i = blocks * groupSize; i < strideCount; i++)
			{
				reedSolomon.SetArgs(clSamplesBytes, clEncodeGx0, clParity0, clParity1, this.npar, i);
				openCLCQ.EnqueueNDRangeKernel(reedSolomon, groupSize, npar / groupSize);
				var temp = clParity0; clParity0 = clParity1; clParity1 = temp;
			}

			//openCLCQ.EnqueueReadBuffer(clRiceOutput, false, 0, (channels * frameSize * (writer.PCM.BitsPerSample + 1) + 256) / 8 * frameCount, clRiceOutputPtr);
		}
	}

#if LKJLKJLJK
    public static class OpenCLExtensions
    {
        public static void SetArgs(this Kernel kernel, params object[] args)
        {
            int i = 0;
            foreach (object arg in args)
            {
                if (arg is int)
                    kernel.SetArg(i, (int)arg);
                else if (arg is Mem)
                    kernel.SetArg(i, (Mem)arg);
                else
                    throw new ArgumentException("Invalid argument type", arg.GetType().ToString());
                i++;
            }
        }

        public static void EnqueueNDRangeKernel(this CommandQueue queue, Kernel kernel, long localSize, long globalSize)
        {
            if (localSize == 0)
                queue.EnqueueNDRangeKernel(kernel, 1, null, new long[] { globalSize }, null);
            else
                queue.EnqueueNDRangeKernel(kernel, 1, null, new long[] { localSize * globalSize }, new long[] { localSize });
        }

        public static void EnqueueNDRangeKernel(this CommandQueue queue, Kernel kernel, long localSizeX, long localSizeY, long globalSizeX, long globalSizeY)
        {
            queue.EnqueueNDRangeKernel(kernel, 2, null, new long[] { localSizeX * globalSizeX, localSizeY * globalSizeY }, new long[] { localSizeX, localSizeY });
        }
    }
#endif
}
