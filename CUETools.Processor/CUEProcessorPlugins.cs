using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using CUETools.Codecs;
using CUETools.Compression;
using CUETools.Ripper;

namespace CUETools.Processor
{
    public static class CUEProcessorPlugins
    {
        public static List<Type> encs;
        public static List<Type> decs;
        public static List<Type> arcp;
        public static List<string> arcp_fmt;
        public static Type hdcd;
        public static Type ripper;

        static CUEProcessorPlugins()
        {
            encs = new List<Type>();
            decs = new List<Type>();
            arcp = new List<Type>();
            arcp_fmt = new List<string>();

            encs.Add(typeof(CUETools.Codecs.WAVWriter));
            decs.Add(typeof(CUETools.Codecs.WAVReader));

            //ApplicationSecurityInfo asi = new ApplicationSecurityInfo(AppDomain.CurrentDomain.ActivationContext);
            //string arch = asi.ApplicationId.ProcessorArchitecture;
            //ActivationContext is null most of the time :(

            string plugins_path = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "plugins");
            if (Directory.Exists(plugins_path))
            {
                AddPluginDirectory(plugins_path);
                string arch = Type.GetType("Mono.Runtime", false) != null ? "mono" : Marshal.SizeOf(typeof(IntPtr)) == 8 ? "x64" : "win32";
                plugins_path = Path.Combine(plugins_path, arch);
                if (Directory.Exists(plugins_path))
                    AddPluginDirectory(plugins_path);
            }
        }

        private static void AddPluginDirectory(string plugins_path)
        {
            foreach (string plugin_path in Directory.GetFiles(plugins_path, "CUETools.*.dll", SearchOption.TopDirectoryOnly))
            {
                try
                {
                    AddPlugin(plugin_path);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(ex.Message);
                }
            }
        }
  
        private static void AddPlugin(string plugin_path)
        {
            AssemblyName name = AssemblyName.GetAssemblyName(plugin_path);
            Assembly assembly = Assembly.Load(name);
            System.Diagnostics.Trace.WriteLine("Loaded " + assembly.FullName);
            foreach (Type type in assembly.GetExportedTypes())
            {
                try
                {
                    if (Attribute.GetCustomAttribute(type, typeof(AudioDecoderClass)) != null)
                    {
                        decs.Add(type);
                    }
                    //if (type.IsClass && !type.IsAbstract && typeof(IAudioDest).IsAssignableFrom(type))
                    if (Attribute.GetCustomAttributes(type, typeof(AudioEncoderClassAttribute)).Length > 0)
                    {
                        encs.Add(type);
                    }
                    CompressionProviderClass archclass = Attribute.GetCustomAttribute(type, typeof(CompressionProviderClass)) as CompressionProviderClass;
                    if (archclass != null)
                    {
                        arcp.Add(type);
                        if (!arcp_fmt.Contains(archclass.Extension))
                            arcp_fmt.Add(archclass.Extension);
                    }
                    if (type.Name == "HDCDDotNet")
                    {
                        hdcd = type;
                    }
                    if (type.IsClass && !type.IsAbstract && typeof(ICDRipper).IsAssignableFrom(type))
                    {
                        ripper = type;
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(ex.Message);
                }
            }
        }
    }
}
