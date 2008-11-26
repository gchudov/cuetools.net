using System;
using System.Collections.Generic;
using System.Collections;
using System.Text;
using System.IO;
using System.Security;
using Microsoft.Win32;

namespace Bwg.Hardware
{
    /// <summary>
    /// This class is the manager that reads the registry and determines which devices exist
    /// </summary>
    public class HardwareManager : IEnumerable<HardwareDevice>, IEnumerable
    {
        #region private member variables
        private IList<HardwareDevice> m_devices;
        #endregion

        #region constructors
        /// <summary>
        /// The constructor for the device manager, creates a list for devices to be added.
        /// </summary>
        public HardwareManager()
        {
            m_devices = new List<HardwareDevice>();
        }
        #endregion

        /// <summary>
        /// Return an enumerator for iterating over all of the tracks on the
        /// disk.
        /// </summary>
        /// <returns>iterator</returns>
        public IEnumerator<HardwareDevice> GetEnumerator()
        {
            return m_devices.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return m_devices.GetEnumerator();
        }

        /// <summary>
        /// Open a given machine, or the default machine if no machine is given
        /// </summary>
        /// <param name="keyname"></param>
        /// <returns></returns>
        public bool OpenMachine(string keyname)
        {
            if (keyname == string.Empty)
                keyname = "System\\CurrentControlSet\\Enum";

            RegistryKey key = Registry.LocalMachine.OpenSubKey(keyname);
            if (key == null)
                return false;

            CheckForHardware(key);
            key.Close();
            return true;
        }

        /// <summary>
        /// Find a specific device given its class, vendor, and product
        /// </summary>
        /// <param name="classname">the class of the device</param>
        /// <param name="vendor">the vendor id of the device</param>
        /// <param name="product">the product id of the device</param>
        /// <returns></returns>
        public HardwareDevice FindDevice(string classname, string vendor, string product)
        {
            foreach (HardwareDevice dev in m_devices)
            {
                if (dev.Class != classname)
                    continue;

                if (dev.Name.Contains(vendor) && dev.Name.Contains(product))
                {
                    if (dev.IsUsb && dev.Location.Length == 0)
                        dev.Location = FindUsbLocation(dev);

                    return dev;
                }
            }
            return null;
        }

        #region private methods

        private string FindUsbLocation(HardwareDevice cddev)
        {
            string result = string.Empty;
            int index = cddev.KeyName.LastIndexOf('&');
            if (index == -1)
                return result;

            string devname = cddev.KeyName.Substring(0, index);
            foreach (HardwareDevice dev in m_devices)
            {
                if (dev.Class == "USB" && dev.KeyName == devname)
                    return dev.Location;
            }

            return result;
        }

        private static bool FindClassString(string s1)
        {
            return s1 == "Class";
        }

        private static bool FindClassGUIDString(string s1)
        {
            return s1 == "ClassGUID";
        }

        private void CreateDevice(RegistryKey key)
        {
            string keyname = key.ToString();
            string onekey ;
            HardwareDevice dev;

            try
            {
                onekey = Path.GetFileName(keyname);
            }
            catch (ArgumentException)
            {
                return;
            }

            dev = new HardwareDevice(onekey);
            object value;

            value = key.GetValue("Class");
            if (value != null)
            {
                if (value is string)
                    dev.Class = (string)value;
                else if (value is string[])
                {
                    dev.Class = ((string[])value)[0];
                }
            }

            value = key.GetValue("ClassGUID");
            if (value != null && value is string)
                dev.ClassGUID = new Guid((string)value);

            value = key.GetValue("DeviceDesc");
            if (value != null && value is string)
                dev.Description = (string)value;

            value = key.GetValue("FriendlyName");
            if (value != null && value is string)
                dev.Name = (string)value;

            value = key.GetValue("LocationInformation");
            if (value != null && value is string)
                dev.Location = (string)value;

            value = key.GetValue("HardwareID");
            if (value != null)
            {
                if (value is string)
                    dev.AddHardware((string)value);
                else if (value is string[])
                {
                    foreach (string str in (string[])value)
                        dev.AddHardware(str);
                }
            }

            m_devices.Add(dev);
        }

        private void CheckForHardware(RegistryKey key)
        {
            string[] values = key.GetValueNames();
            if (Array.Find(values, FindClassString) != null && Array.Find(values, FindClassGUIDString) != null)
            {
                // This is a device
                CreateDevice(key);
            }
            else
            {
                values = key.GetSubKeyNames();
                foreach (string str in values)
                {
                    try
                    {
                        RegistryKey subkey = key.OpenSubKey(str);
                        if (subkey != null)
                        {
                            CheckForHardware(subkey);
                            subkey.Close();
                        }
                    }
                    catch (SecurityException)
                    {
                    }
                }
            }
        }

        #endregion
    }
}
