using System;
using System.Collections.Generic;
using System.Text;

namespace Bwg.Scsi
{
    /// <summary>
    /// This class contains information about a burner device that is useful to have
    /// even when the device it not open
    /// </summary>
    public class DeviceInfo
    {
        #region private member variables
        /// <summary>
        /// The device name for the device
        /// </summary>
        private string m_device_name;

        /// <summary>
        /// The device latter for the device
        /// </summary>
        private string m_device_letter;

        /// <summary>
        /// The result of an inquiry
        /// </summary>
        private InquiryResult m_inqury_result;

        #endregion

        #region static private member variables
        /// <summary>
        /// The list of devices found and created.  For each device only a single DeviceInfo object
        /// will ever be created and this list holds the objects created to date
        /// </summary>
        private static IList<DeviceInfo> m_devices = new List<DeviceInfo>();
        #endregion

        #region constructor
        /// <summary>
        /// Create a new device info object given the device name and drive letter for the device
        /// </summary>
        /// <param name="name">the device name</param>
        /// <param name="letter">the drive letter</param>
        private DeviceInfo(string name, string letter)
        {
            m_device_name = name;
            m_device_letter = letter;
        }
        #endregion

        #region public properties
        /// <summary>
        /// This property returns the drive letter associated with the device
        /// </summary>
        public string DeviceLetter
        {
            get { return m_device_letter; }
        }

        /// <summary>
        /// This property returns the NT device name associated with the device
        /// </summary>
        public string DeviceName
        {
            get { return m_device_name; }
        }

        /// <summary>
        /// This property returns a short description of the drive including drive letter
        /// and NT device name.
        /// </summary>
        public string ShortDesc
        {
            get { return m_device_letter + " ( " + m_device_name + " )"; }
        }

        /// <summary>
        /// This property return a long description of the drive including the drive letter,
        /// NT device name, vendor ID and product ID.
        /// </summary>
        public string LongDesc
        {
            get
            {
                string result = m_device_letter + " ";
                // result += " ( " + m_device_name + " ) ";
                result += "[" + m_inqury_result.VendorIdentification + " " +
                    m_inqury_result.ProductIdentification + " " +
                    m_inqury_result.FirmwareVersion + "]";
                return result;
            }
        }
        
        /// <summary>
        /// This property returns the inquiry data from the inquiry request
        /// </summary>
        public InquiryResult InquiryData
        {
            get { return m_inqury_result; }
        }
        #endregion

        #region public methods

        /// <summary>
        /// This method returns a string representation of this object
        /// </summary>
        /// <returns>string representatation of the object</returns>
        public override string ToString()
        {
            return m_device_letter + " (" + m_inqury_result.VendorIdentification.Trim() + " " + m_inqury_result.ProductIdentification.Trim() + ")";
        }

        /// <summary>
        /// Extract the information we want to keep around after we close the device.
        /// </summary>
        /// <param name="dev">the open device we are going to query</param>
        /// <returns>true if we got the info, false otherwise</returns>
        public bool ExtractInfo(Device dev)
        {
            if (dev.Inquiry(out m_inqury_result) != Device.CommandStatus.Success)
                return false;

            if (!m_inqury_result.Valid)
                return false;

            if (m_inqury_result.PeripheralQualifier != 0)
                return false;

            if (m_inqury_result.PeripheralDeviceType != Device.MMCDeviceType)
                return false;

            return true;
        }
        #endregion

        #region public static methods
        /// <summary>
        /// This method returns or creates a unique device info structure based on the name and the
        /// drive letter for a given drive.
        /// </summary>
        /// <param name="name">the NT name of the drive</param>
        /// <param name="letter">the drive letter for the drive</param>
        /// <returns>the single DeviceInfo object that represents this drive</returns>
        public static DeviceInfo CreateDevice(string name, string letter)
        {
            foreach (DeviceInfo info in m_devices)
            {
                if (info.DeviceName == name && info.DeviceLetter == letter)
                    return info;
            }

            DeviceInfo newdev = new DeviceInfo(name, letter);
            m_devices.Add(newdev);
            return newdev;
        }
        #endregion
    }
}
