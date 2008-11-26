//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for Windows XP
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
using System.Diagnostics;

namespace Bwg.Scsi
{
    /// <summary>
    /// This type represents a location on a disk in minute, seconds, frame format
    /// </summary>
    [Serializable()]
    public class MinuteSecondFrame
    {
        /// <summary>
        /// The number of 44100 Hz samples in a single frame
        /// </summary>
        public const long SamplesPerFrame = 588;

        /// <summary>
        /// The number of frames in a second
        /// </summary>
        public const long FramesPerSecond = 75;

        /// <summary>
        /// The number of seconds in a minute.
        /// </summary>
        public const long SecondsPerMinute = 60;

        /// <summary>
        /// The minutes of this time frame
        /// </summary>
        public int Minutes;

        /// <summary>
        /// The seconds of this time frame
        /// </summary>
        public byte Seconds;

        /// <summary>
        /// The frames (at FramesPerSecond per second) of this time frame
        /// </summary>
        public byte Frames;

        /// <summary>
        /// Constructor for the type
        /// </summary>
        /// <param name="m"></param>
        /// <param name="s"></param>
        /// <param name="f"></param>
        public MinuteSecondFrame(int m, int s, int f)
        {
            Minutes = m;
            Seconds = (byte)s;
            Frames = (byte)f;
        }

        /// <summary>
        /// Create a MinuteSecondFrame value from a sector count
        /// </summary>
        /// <param name="sectors"></param>
        public MinuteSecondFrame(long sectors)
        {
            long m = sectors / (FramesPerSecond * SecondsPerMinute);
            sectors -= m * FramesPerSecond * SecondsPerMinute;

            long s = sectors / FramesPerSecond;
            sectors -= s * FramesPerSecond;

            Debug.Assert(s < SecondsPerMinute);
            Debug.Assert(sectors < FramesPerSecond);

            Minutes = (int)m;
            Seconds = (byte)s;
            Frames = (byte)sectors;
        }

        /// <summary>
        /// Return the sector count for this time span
        /// </summary>
        public long SectorCount
        {
            get
            {
                return Minutes * SecondsPerMinute * FramesPerSecond + Seconds * FramesPerSecond + Frames;
            }
        }

        /// <summary>
        /// This property returns the logical block address associated with the time given.
        /// </summary>
        public long LogicalBlockAddress
        {
            get
            {
                long result = Minutes * FramesPerSecond * SecondsPerMinute + Seconds * FramesPerSecond + Frames;

                if (Minutes < 90)
                    result -= 150;
                else
                    result -= 450150;

                return result;
            }
        }


        /// <summary>
        /// Return a hash code for this object
        /// </summary>
        /// <returns>hash value</returns>
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        /// <summary>
        /// Returns true if the two objects are equal
        /// </summary>
        /// <param name="o">the object to compare to</param>
        /// <returns>true if the objects are equal, false otherwise</returns>
        public override bool Equals(Object o)
        {
            if (o == null)
                return false;

            if (!(o is MinuteSecondFrame))
                return false;


            MinuteSecondFrame other = (MinuteSecondFrame)o;
            return other.Minutes == Minutes && other.Seconds == Seconds && other.Frames == Frames;
        }

        /// <summary>
        /// Convert to a string
        /// </summary>
        public override string ToString()
        {
            return Minutes.ToString("") + ":" + Seconds.ToString("") + "." + Frames.ToString("") + ", " + SectorCount + " sectors";
        }

        /// <summary>
        /// convert to a string with format options
        /// </summary>
        /// <param name="fmt">the format to use for the output</param>
        /// <returns>a string representing the object</returns>
        public string ToString(string fmt)
        {
            string result = "";

            if (fmt == "M:S")
            {
                result = Minutes.ToString("d2") + ":" + Seconds.ToString("d2");
            }
            else if (fmt == "M:S.F")
            {
                result = Minutes.ToString("d2") + ":" + Seconds.ToString("d2") + "." + Frames.ToString("d2");
            }
			else if (fmt == "M:S:F")
			{
				result = Minutes.ToString("d2") + ":" + Seconds.ToString("d2") + ":" + Frames.ToString("d2");
			}
			else
            {
                throw new FormatException("the format '" + fmt + "' is not valid for the class Bwg.Scsi.MinuteSecondFrame");
            }

            return result;
        }

        /// <summary>
        /// Convert a number of samples into an MSF object
        /// </summary>
        /// <param name="samples">the number of samples</param>
        /// <returns></returns>
        static public MinuteSecondFrame FromSampleCount(long samples)
        {
            long frames = samples / SamplesPerFrame;
            uint m = (uint)(frames / (FramesPerSecond * SecondsPerMinute));
            frames -= m * FramesPerSecond * SecondsPerMinute;

            uint s = (uint)(frames / FramesPerSecond);
            frames -= s * FramesPerSecond;

            Debug.Assert(s < SecondsPerMinute);
            Debug.Assert(m < FramesPerSecond);
            return new MinuteSecondFrame((byte)m, (byte)s, (byte)frames);
        }

        /// <summary>
        /// Operator for adding two MinuteSecondFrame times together
        /// </summary>
        /// <param name="t1">the first operand</param>
        /// <param name="t2">the second operand</param>
        /// <returns>the sum of t1 and t2</returns>
        static public MinuteSecondFrame operator +(MinuteSecondFrame t1, MinuteSecondFrame t2)
        {
            return new MinuteSecondFrame(t1.SectorCount + t2.SectorCount);
        }

        /// <summary>
        /// Operator for adding a time value with a frame count (or sector count)
        /// </summary>
        /// <param name="t1">the time value</param>
        /// <param name="t2">the count in sectors (or frames)</param>
        /// <returns>the addition time</returns>
        static public MinuteSecondFrame operator +(MinuteSecondFrame t1, long t2)
        {
            return new MinuteSecondFrame(t1.SectorCount + t2);
        }

        /// <summary>
        /// Returns true if the first object is less than the second object
        /// </summary>
        /// <param name="t1">the first object</param>
        /// <param name="t2">the second object</param>
        /// <returns>true if t1 is less than t2 </returns>
        static public bool operator <(MinuteSecondFrame t1, MinuteSecondFrame t2)
        {
            if (t1.Minutes < t2.Minutes)
                return true;

            if (t1.Minutes > t2.Minutes)
                return false;

            if (t1.Seconds < t2.Seconds)
                return true;

            if (t1.Seconds > t2.Seconds)
                return false;

            if (t1.Frames < t2.Frames)
                return true;

            return false;
        }

        /// <summary>
        /// Returns true if the first object is greater than the second object
        /// </summary>
        /// <param name="t1">the first object</param>
        /// <param name="t2">the second object</param>
        /// <returns>true, if t1 is greater than t2</returns>
        static public bool operator >(MinuteSecondFrame t1, MinuteSecondFrame t2)
        {
            if (t1.Minutes > t2.Minutes)
                return true;

            if (t1.Minutes < t2.Minutes)
                return false;

            if (t1.Seconds > t2.Seconds)
                return true;

            if (t1.Seconds < t2.Seconds)
                return false;

            if (t1.Frames > t2.Frames)
                return true;

            return false;
        }

        /// <summary>
        /// returns true if t1 is less than or equal to t2
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <returns></returns>
        static public bool operator <=(MinuteSecondFrame t1, MinuteSecondFrame t2)
        {
            return (t1 < t2) || (t1 == t2);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="t1"></param>
        /// <param name="t2"></param>
        /// <returns></returns>
        static public bool operator >=(MinuteSecondFrame t1, MinuteSecondFrame t2)
        {
            return (t1 > t2) || (t1 == t2);
        }
    }
}
