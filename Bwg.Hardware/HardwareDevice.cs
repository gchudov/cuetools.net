using System;
using System.Collections.Generic;
using System.Text;

namespace Bwg.Hardware
{
    /// <summary>
    /// This class represents a hardware device as extracted from the registry.
    /// </summary>
    public class HardwareDevice
    {
        #region private member variables
        private string m_name;
        private string m_desc;
        private string m_class;
        private Guid m_class_guid;
        private string m_key_name ;
        private string m_location;
        private string[] m_hardware;
        #endregion

        #region constructor
        /// <summary>
        /// The constructor for a device.
        /// </summary>
        /// <param name="key">the name of the registry key that contains info about this device</param>
        public HardwareDevice(string key)
        {
            m_key_name = key;
            m_name = string.Empty ;
            m_desc = string.Empty ;
            m_location = string.Empty;

            m_hardware = new string[0];
        }
        #endregion

        #region public methods
        /// <summary>
        /// Add a new hardware address to the device
        /// </summary>
        /// <param name="addr">the new address to add to the device</param>
        public void AddHardware(string addr)
        {
            int count = m_hardware.GetLength(0);
            Array.Resize(ref m_hardware, count + 1);
            m_hardware[count] = addr;
        }
        #endregion

        #region public properties

        /// <summary>
        /// The name of the key where this information was extracted
        /// </summary>
        public string KeyName
        {
            get { return m_key_name; }
        }

        /// <summary>
        /// This property is the (friendly) name of the device
        /// </summary>
        public string Name
        {
            get
            {
                if (m_name != string.Empty)
                    return m_name;

                if (m_desc != string.Empty)
                    return m_desc;

                return m_key_name;
            }
            set
            {
                m_name = value;
            }
        }

        /// <summary>
        /// This property is the description of the device
        /// </summary>
        public string Description
        {
            get
            {
                return m_desc;
            }
            set
            {
                m_desc = value;
            }
        }

        /// <summary>
        /// This property is the class that the device belong to
        /// </summary>
        public string Class
        {
            get
            {
                return m_class;
            }
            set
            {
                m_class = value;
            }
        }

        /// <summary>
        /// This property is the GUID for the class that the device belongs to
        /// </summary>
        public Guid ClassGUID
        {
            get
            {
                return m_class_guid;
            }
            set
            {
                m_class_guid = value;
            }
        }

        /// <summary>
        /// This property is the location of the device
        /// </summary>
        public string Location
        {
            get { return m_location; }
            set { m_location = value; }
        }

        /// <summary>
        /// This class is the hardware addresses assocaited with the device
        /// </summary>
        public string[] Hardware
        {
            get
            {
                return m_hardware;
            }
        }

        /// <summary>
        /// This property returns true if the device is an IDE device
        /// </summary>
        public bool IsIde
        {
            get
            {
                return HardwareStartsWith("IDE");
            }
        }

        /// <summary>
        /// This property returns true if this is a USB device
        /// </summary>
        public bool IsUsb
        {
            get
            {
                return HardwareStartsWith("USB") ;
            }
        }

        /// <summary>
        /// This property returns true if this is a Firewire device
        /// </summary>
        public bool IsFirewire
        {
            get
            {
                return HardwareStartsWith("SBP2");
            }
        }

        /// <summary>
        /// This property returns true if this is a SCSI device
        /// </summary>
        public bool IsScsi
        {
            get
            {
                return HardwareStartsWith("SCSI");
            }
        }
        #endregion

        #region private methods
        /// <summary>
        /// This method returns true if the hardware address start with the string given
        /// </summary>
        /// <param name="begin">the string to start with</param>
        /// <returns>true if the string begins with the right value, otherwise false</returns>
        public bool HardwareStartsWith(string begin)
        {
            foreach (string str in m_hardware)
            {
                if (str.StartsWith(begin))
                    return true;
            }
            return false;
        }
        #endregion
    }
}
