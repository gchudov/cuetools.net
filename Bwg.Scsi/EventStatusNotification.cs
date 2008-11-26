//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for DotNet
// 
// Copyright (C) 2006 by Jack W. Griffin (butchg@comcast.net)
//
// This program is free software; you can redistribute it and/or modify 
// it under the terms of the GNU General Public License as published by 
// the Free Software Foundation; either version 2 of the License, or 
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
// for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program; if not, write to the 
//
// Free Software Foundation, Inc., 
// 59 Temple Place, Suite 330, 
// Boston, MA 02111-1307 USA
//

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace Bwg.Scsi
{
    /// <summary>
    /// This class contains information about event status communicated from the device
    /// via the GetEventStatusNotification SCSI command.
    /// </summary>
    public class EventStatusNotification : Result
    {
        #region public types

        /// <summary>
        /// This field indicates the type of event
        /// </summary>
        public enum EventType
        {
            /// <summary>
            /// No event was returned
            /// </summary>
            NoEvent = 0x00,

            /// <summary>
            /// There was an operation change
            /// </summary>
            OperationalChange = 0x01,

            /// <summary>
            /// There was a change in power management status
            /// </summary>
            PowerManagement = 0x02,

            /// <summary>
            /// There was an external request to the device
            /// </summary>
            ExternalRequest = 0x03,

            /// <summary>
            /// The status of the media has changed
            /// </summary>
            Media = 0x04,

            /// <summary>
            /// There was a change in the multiple initiator device
            /// </summary>
            MultipleInitiator = 0x05,

            /// <summary>
            /// There was a change in the busy state of the device
            /// </summary>
            DeviceBusy = 0x06,

            /// <summary>
            /// This value is reserved and should not be returned
            /// </summary>
            Reserved = 0x07,
        } ;
        #endregion

        #region public member variables

        /// <summary>
        /// This property returns true if an event is available
        /// </summary>
        public readonly bool EventAvailable;

        /// <summary>
        /// This property is a byte that contains a mask of the supported event classes
        /// </summary>
        public readonly byte SupportedEventClasses;

        /// <summary>
        /// This property contains the notification class of the specific event that has
        /// occurred.
        /// </summary>
        public readonly EventType NotificationClass;

        /// <summary>
        /// This array contains the specific data returned describing the event.  The format of this
        /// data is described in the SCSI-3 MMC specification.
        /// </summary>
        public readonly byte [] EventData;
        #endregion

        #region constructor
        /// <summary>
        /// This is the contructor for this class which contains information about any events that
        /// have occurred in the SCSI device.
        /// </summary>
        /// <param name="buffer">Pointer to a memory area containing the reply from the SCSI device</param>
        /// <param name="size">The size of the memory area containing the reply from the SCSI device</param>
        public EventStatusNotification(IntPtr buffer, int size) : base(buffer, size)
        {
            SupportedEventClasses = Marshal.ReadByte(buffer, 3);

            byte b = Marshal.ReadByte(buffer, 2);
            if ((b & 0x80) != 0)
            {
                EventAvailable = false;
            }
            else
            {
                ushort len = Get16(0);
                EventAvailable = true;
                NotificationClass = (EventType)(b & 0x03);
                EventData = new byte[len];
                for (int i = 0; i < len; i++)
                    EventData[i] = Marshal.ReadByte(buffer, 4 + i);
            }
        }
        #endregion

        #region public properties
        /// <summary>
        /// This property returns true if Operational Events are supported on this device
        /// </summary>
        public bool OperationalChangeEventsSupported { get { return (SupportedEventClasses & 0x02) != 0; } }

        /// <summary>
        /// This property returns true if Power Management Events are supported on this device
        /// </summary>
        public bool PowerManagementEventsSupported { get { return (SupportedEventClasses & 0x04) != 0; } }

        /// <summary>
        /// This property returns true if External Request Events are supported on this device
        /// </summary>
        public bool ExternalRequestEventsSupported { get { return (SupportedEventClasses & 0x08) != 0; } }

        /// <summary>
        /// This property returns true if Media Events are supported on this device
        /// </summary>
        public bool MediaEventsSupported { get { return (SupportedEventClasses & 0x10) != 0; } }

        /// <summary>
        /// This property returns true if Multiple Initiator Events are supported on this device
        /// </summary>
        public bool MultiInitiatorEventsSupported { get { return (SupportedEventClasses & 0x20) != 0; } }

        /// <summary>
        /// This property returns true if Device Busy Events are supported on this device
        /// </summary>
        public bool DeviceBusyEventsSupported { get { return (SupportedEventClasses & 0x40) != 0; } }
        #endregion
    }
}
