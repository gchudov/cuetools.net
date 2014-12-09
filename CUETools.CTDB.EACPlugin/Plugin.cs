using System;
using System.Collections.Generic;
using System.Text;
using HelperFunctionsLib;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Net;
using System.IO;
using CUETools.CDImage;
using CUETools.AccurateRip;
using CUETools.Codecs;
using CUETools.CTDB;
using CUETools.CTDB.EACPlugin;

namespace AudioDataPlugIn
{
    [Guid("C02A1BF2-5C46-4990-80C2-78E8C395CB80"),
    ClassInterface(ClassInterfaceType.None),
    ComSourceInterfaces(typeof(IAudioDataTransfer)),
    ]

    // Our class needs to inherit the IAudioDataTransfer in
    // order to be found by EAC, further the class needs
    // to be named AudioDataTransfer and must be in the
    // namespace AudioDataPlugIn

    public class AudioDataTransfer : IAudioDataTransfer
    {
        int m_start_pos = 0, m_length = 0;
        bool m_test_mode = false;
        IMetadataLookup m_data = null;
        CDImageLayout TOC;
#if USEAR
		string ArId;
#endif
        AccurateRipVerify ar;
        AccurateRipVerify arTest;
        CUEToolsDB ctdb;
        bool sequence_ok = true;
        bool is_secure_mode;
        bool is_offset_set;
        string m_drivename;
#if DEBUG
        StringWriter m_trace;
#endif


        // This functions just returns an unique identifier.
        // For that, the string representation of the unique
        // guid is used
        public string GetAudioTransferPluginGuid()
        {
            return ((GuidAttribute)Attribute.GetCustomAttribute(GetType(), typeof(GuidAttribute))).Value;
        }


        // Each plugin has also an (unique) display name
        // which will be used for selecting/deselecting
        // the plugin and for display in the log file
        public string GetAudioTransferPluginName()
        {
            return "CUETools DB Plugin V2.1.6";
        }

        // Each plugin should have its own options page.
        // Even though if there are no options, a help or
        // information dialog should be displayed
        public void ShowOptions()
        {
            Options opt = new Options();
            opt.ShowDialog();
        }



        // Now to the audio transfer functions, the sequence how
        // the functions are called is:
        // StartNewSession
        // StartNewTransfer
        // TransferAudio
        // ...
        // TransferAudio
        // TransferFinshed
        // Then perhaps repeating StartNewTransfer to TransferFinished
        // (e.g. when extracting several tracks), and finally
        // EndOfSession
        // This is called just before the log window will be
        // shown. You can return a log output in that stage (or
        // even display a window of your own - even though it should
        // not annoy the user)

        // StartNewSession is called once at the very beginning of an
        // extraction session. It receives the CD metadata, the
        // name of the used drive, the used read offset and whether
        // the offset was setted by AccurateRip (so having a comparable
        // offset value)

        public void StartNewSession(IMetadataLookup data, string drivename, int offset, bool aroffset, int mode)
        {
            // Copy the CD metadata to the object
            m_data = data;

#if DEBUG
            m_trace = new StringWriter();
#endif

            var parts = drivename.Split(' ');
            m_drivename = parts[0].PadRight(8, ' ') + " -";
            for (int i = 1; i < parts.Length; i++)
                m_drivename += " " + parts[i];

            TOC = new CDImageLayout();
            for (int i = 0; i < m_data.NumberOfTracks; i++)
            {
                uint start = m_data.GetTrackStartPosition(i);
                uint next = m_data.GetTrackEndPosition(i);
                TOC.AddTrack(new CDTrack(
                    (uint)i + 1,
                    start,
                    next - start,
                    !m_data.GetTrackDataTrack(i),
                    m_data.GetTrackPreemphasis(i)));
            }
            TOC[1][0].Start = 0U;
            ar = new AccurateRipVerify(TOC, null);
            arTest = new AccurateRipVerify(TOC, null);
            ctdb = new CUEToolsDB(TOC, null);
#if USEAR
			ArId = AccurateRipVerify.CalculateAccurateRipId(TOC);
			ar.ContactAccurateRip(ArId);
#endif
            ctdb.Init(ar);
            this.sequence_ok = true;
            this.m_start_pos = 0;
            this.m_length = 0;
            this.m_test_mode = false;
            this.is_offset_set = aroffset;
            this.is_secure_mode = mode >= 2;
        }

        // This function will be called once per session. A session
        // is e.g. a file on a real extraction (or the equivalent
        // for test extractions). It receives the sector startpos
        // the length in sectors and whether the extraction is performed
        // in test mode

        public void StartNewTransfer(int startpos, int length, bool testmode)
        {
            // Copy the current parameters to the objects variables
            m_start_pos = startpos - (int)TOC[TOC.FirstAudio][0].Start;
            m_length = length;
            m_test_mode = testmode;
            if (this.sequence_ok)
            {
                var thisAr = m_test_mode ? arTest : ar;
                if (this.m_start_pos * 588 != thisAr.Position)
                {
                    if (thisAr.Position == 0 && this.m_start_pos == (int)TOC[TOC.FirstAudio].Pregap)
                    {
#if DEBUG
                        m_trace.WriteLine("Adding pregap");
#endif
                        var ad = new byte[588 * 4];
                        AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, ad, ad.Length / 4);
                        for (int i = 0; i < (int)TOC[TOC.FirstAudio].Pregap; i++)
                            thisAr.Write(buff);
                    }
                    else
                    {
#if DEBUG
                        m_trace.WriteLine("Sequence broken on new trasfer");
#endif
                        this.sequence_ok = false;
                    }
                }
            }
        }


        // This function received the extracted (and 
        // uncompressed/unmodified audio data), but no WAV 
        // header. If you want to write out the WAV file
        // you need to generate one yourself. It will be always
        // 44.1 kHz, stereo 16 bit samples (so 4 bytes per
        // stereo sample)

        public void TransferAudioData(Array audiodata)
        {
            if (this.sequence_ok)
            {
                byte[] ad = (byte[])audiodata;
                AudioBuffer buff = new AudioBuffer(AudioPCMConfig.RedBook, ad, ad.Length / 4);
                var thisAr = m_test_mode ? arTest : ar;
                thisAr.Write(buff);
            }
        }



        // This function is called after a transfer has finished.
        // We don't do here anything, because a track can be delivered
        // in several transfers (index extraction) and as AccurateRip
        // is only (full) track based, we don't do anything...

        public void TransferFinished()
        {
            if (this.sequence_ok)
            {
                var thisAr = m_test_mode ? arTest : ar;
                if ((m_start_pos + m_length) * 588 != thisAr.Position)
                {
#if DEBUG
                    m_trace.WriteLine("Sequence broken on end of trasfer");
#endif
                    this.sequence_ok = false;
                }
            }
        }


        // The extraction has finished, the log dialog will
        // be shown soon, so we can return a string which will
        // be displayed in the log window and be written to
        // the log file. Anyway, you could also return just
        // an empty string, in that case no log output will be done!
        public string EndOfSession()
        {
            StringWriter sw = new StringWriter();
            if (this.sequence_ok)
            {
                if (TOC.AudioLength * 588 != ar.Position)
                {
#if DEBUG
                    m_trace.WriteLine("Sequence broken on end of session");
#endif
                    this.sequence_ok = false;
                }
                if (ar.Position != arTest.Position && arTest.Position > 0)
                {
#if DEBUG
                    m_trace.WriteLine("Sequence broken on end of session");
#endif
                    this.sequence_ok = false;
                }
            }
            if (!this.sequence_ok)
            {
#if DEBUG
                return m_trace.ToString();
#else
                return "";
#endif
            }
#if DEBUG
            sw.Write(m_trace.ToString());
#endif
            if (this.sequence_ok)
            {
#if USEAR
				int conf = (int)ar.WorstConfidence() + 1;
#else
                int conf = 1;
#endif
                var form = new FormSubmitParity(
                    ctdb,
                    "EAC" + m_data.HostVersion + " CTDB 2.1.6",
                    m_drivename,
                    conf,
                    (arTest.Position == 0 && this.is_secure_mode) || (arTest.Position != 0 && arTest.CRC32(0) == ar.CRC32(0)) ? 100 : 0,
                    m_data.AlbumArtist,
                    m_data.AlbumTitle);
                form.ShowDialog();
                sw.WriteLine("[CTDB TOCID: {0}] {1}",
                    TOC.TOCID,
                    ctdb.DBStatus ?? "found");
                if (ctdb.SubStatus != null)
                    sw.WriteLine("Submit result: " + ctdb.SubStatus);
#if DEBUG
                ctdb.GenerateLog(sw, true);
#else
                ctdb.GenerateLog(sw, false);
#endif
                int fixConf = -1;
                int myConf = 0;
                if (ctdb.QueryExceptionStatus == WebExceptionStatus.Success)
                {
                    foreach (DBEntry entry in ctdb.Entries)
                        if (!entry.hasErrors)
                            myConf += entry.conf;
                        else if (entry.canRecover)
                            fixConf = Math.Max(fixConf, entry.conf);
                }
                if (fixConf > myConf)
                    sw.WriteLine("If you are sure that your rip contains errors, you can use CUETools to repair it.");

#if USEAR
                ar.GenerateFullLog(sw, false, ArId);
#endif
            }
            else
                sw.WriteLine("Some tracks have been skipped");
            return sw.ToString();
        }
    }

    //private NativeWindow GetWindowFromHost(int hwnd)
    //{
    //   IntPtr handle = new IntPtr(hwnd);
    //   NativeWindow nativeWindow = new NativeWindow();
    //   nativeWindow.AssignHandle(handle);
    //   return window;
    //}
    //NativeWindow parentWindow = GetWindowFromHwnd(hwnd);
    //try
    //{
    //   launchTarget.ShowDialog(parentWindow);
    //}
    //finally
    //{
    //   parentWindow.ReleaseHandle();
    //}
    //internal class WindowWrapper : System.Windows.Forms.IWin32Window
    //{
    //    public IntPtr Handle { get; private set; }
    //    public WindowWrapper(IntPtr hwnd) { Handle = hwnd; }
    //}
}
