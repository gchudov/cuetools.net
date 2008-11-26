using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices ;
using System.IO;
using Bwg.Logging;

namespace Bwg.Scsi
{
    /// <summary>
    /// This class manages the burner devices that are in the system.
    /// </summary>
    public unsafe class DeviceManager
    {
        #region external functions
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern uint QueryDosDevice(string name, IntPtr buffer, uint size);
        #endregion

        #region private member variables

        /// <summary>
        /// The logger for logging messages
        /// </summary>
        private Logger m_logger;

        /// <summary>
        /// This list contains the device names found in the system
        /// </summary>
        private IList<DeviceInfo> m_devices_found;
        #endregion

        #region constructor
        /// <summary>
        /// This constructor creates a new device manager object.
        /// </summary>
        /// <param name="l">the logger to use for logging messages</param>
        public DeviceManager(Logger l)
        {
            m_logger = l;
            m_devices_found = new List<DeviceInfo>();
        }
        #endregion

        #region public variables
        /// <summary>
        /// This event is fired just before a rescan of the devices in the system.
        /// </summary>
        public EventHandler<DeviceManagerRescanArgs> BeforeScan;

        /// <summary>
        /// This event is triggered just after a rescan of devices in the system.
        /// </summary>
        public EventHandler<DeviceManagerRescanArgs> AfterScan;
        #endregion

        #region public properties
        /// <summary>
        /// This property returns the list of devices found as a result of the most recent
        /// scan.
        /// </summary>
        public IList<DeviceInfo> DevicesFound
        {
            get { return m_devices_found; }
        }

        /// <summary>
        /// This property returns the logger assocaited with the device manager
        /// </summary>
        public Logger MyLogger
        {
            get { return m_logger; }
        }
        #endregion

        #region public methods
        /// <summary>
        /// This method scans the current machine for all CDROM devices found in the system.
        /// </summary>
        public void ScanForDevices()
        {
            if (BeforeScan != null)
            {
                DeviceManagerRescanArgs args = new DeviceManagerRescanArgs();
                BeforeScan(this, args);
            }

            m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 0, "Scanning for devices ... ")) ;

            OperatingSystem os = Environment.OSVersion;
            m_logger.LogMessage(new UserMessage(UserMessage.Category.Info, 0, "Operating System: " + os.ToString()));
            m_logger.LogMessage(new UserMessage(UserMessage.Category.Info, 0, "Platform: " + os.Platform.ToString()));

            int ossize = 32 ;
            if (IntPtr.Size == 8)
                ossize = 64 ;

            m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 0, "OS Word Size: " + ossize.ToString())) ;

            m_devices_found.Clear();
            for (int i = 0; i < 100; i++)
            {
                uint dlev = (uint)((i > 5) ? 9 : 8);
                string name = "\\\\.\\CDROM" + i.ToString();

                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, dlev, "Checking device " + name + " ... "));

                Device dev = new Device(m_logger) ;
                if (!dev.Open(name))
                {
                    m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, dlev, "  ... device open failed"));
                    continue ;
                }

                string letter = GetLetterFromDeviceName(name);

                DeviceInfo info = DeviceInfo.CreateDevice(name, letter);
                if (!info.ExtractInfo(dev))
                {
                    m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, dlev, "  ... cannot extract inquiry information from the drive"));

                    string str = "The drive '" + letter + "' (" + name + ") is a CD/DVD driver, but is not a valid MMC device.";
                    m_logger.LogMessage(new UserMessage(UserMessage.Category.Error, 0, str));
                    str = "This drive is not supported by BwgBurn and is probably an older device.";
                    m_logger.LogMessage(new UserMessage(UserMessage.Category.Error, 0, str));
                    continue;
                }

                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, dlev, "  ... device added to device list"));
                m_devices_found.Add(info) ;
                dev.Close() ;
            }

            if (AfterScan != null)
            {
                DeviceManagerRescanArgs args = new DeviceManagerRescanArgs();
                AfterScan(this, args);
            }

            string devlist = string.Empty;
            foreach (DeviceInfo info in m_devices_found)
            {
                if (devlist.Length > 0)
                    devlist += ", ";
                devlist += info.ShortDesc;
            }
            m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 0, "Found devices ... " + devlist));

            foreach (DeviceInfo info in m_devices_found)
            {
                UserMessage m = new UserMessage(UserMessage.Category.Debug, 6, "Found Device: " + info.DeviceLetter) ;
                m_logger.LogMessage(m);

                m = new UserMessage(UserMessage.Category.Debug, 6, "    NT Name = " + info.DeviceName);
                m_logger.LogMessage(m);

                m = new UserMessage(UserMessage.Category.Debug, 6, "    Vendor = " + info.InquiryData.VendorIdentification.Trim());
                m_logger.LogMessage(m);

                m = new UserMessage(UserMessage.Category.Debug, 6, "    Product = " + info.InquiryData.ProductIdentification.Trim());
                m_logger.LogMessage(m);

                m = new UserMessage(UserMessage.Category.Debug, 6, "    Revision = " + info.InquiryData.ProductRevision.Trim());
                m_logger.LogMessage(m);
            }
        }
        #endregion

        #region private methods

        private static string GetNtDeviceNameForDrive(string name)
        {
            string result;
            const int buflen = 512;
            uint ret;
            byte[] buffer = new byte[buflen];

            if (name.EndsWith("\\"))
                name = name.Substring(0, name.Length - 1);

            IntPtr ptr = Marshal.AllocHGlobal(buflen);
            {
                ret = QueryDosDevice(name, ptr, buflen / 2);
                Marshal.Copy(ptr, buffer, 0, buflen);

                int totallen = buflen;
                for (int i = 0; i < buflen - 2; i++)
                {
                    if (buffer[i] == 0 && buffer[i + 1] == 0)
                    {
                        totallen = i + 1;
                        break;
                    }
                }
                result = Encoding.Unicode.GetString(buffer, 0, totallen);
            }

            return result;
        }

        /// <summary>
        /// Return the drive letter for a drive given its NT name
        /// </summary>
        /// <param name="devname">the NT name</param>
        /// <returns>the drive letter</returns>
        public static string GetLetterFromDeviceName(string devname)
        {
            if (devname.StartsWith("\\\\.\\"))
                devname = devname.Substring(4);

            foreach (DriveInfo info in DriveInfo.GetDrives())
            {
                if (info.DriveType == DriveType.CDRom)
                {
                    string ntname = GetNtDeviceNameForDrive(info.Name);
                    if (ntname.StartsWith("\\Device\\"))
                        ntname = ntname.Substring(8);

                    if (ntname.ToLower() == devname.ToLower())
                        return info.Name;
                }
            }

            return string.Empty;
        }
        #endregion
    }
}
