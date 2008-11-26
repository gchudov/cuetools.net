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
using System.Runtime.InteropServices;

namespace Bwg.Logging
{
    /// <summary>
    /// This class allows for the logging of messages
    /// </summary>
    public class Logger
    {
        /// <summary>
        /// This is used to lock the logger when multiple threads are
        /// trying to log messages concurrently
        /// </summary>
        private Object m_lock_object;

        /// <summary>
        /// This is a map from message type to sinks
        /// </summary>
        private IDictionary<UserMessage.Category, Sink> m_sinks;

        /// <summary>
        /// This is a map from the message type to the level that is being filtered. Anything
        /// below the value found here is not displayed
        /// </summary>
        private IDictionary<UserMessage.Category, int> m_level;

        /// <summary>
        /// Create the logging object
        /// </summary>
        public Logger()
        {
            m_lock_object = new Object();
            m_sinks = new Dictionary<UserMessage.Category, Sink>();
            m_level = new Dictionary<UserMessage.Category, int>();
        }

        /// <summary>
        /// Assocated a sink with a given type of message
        /// </summary>
        /// <param name="c">the category of message</param>
        /// <param name="s">the sink for this category of message</param>
        public void SetSink(UserMessage.Category c, Sink s)
        {
            lock (m_lock_object)
            {
                m_sinks[c] = s;
            }
        }

        /// <summary>
        /// Return the sink given the type of message
        /// </summary>
        /// <param name="c">the type of sink</param>
        /// <returns>the sink associated with the message type</returns>
        public Sink GetSink(UserMessage.Category c)
        {
            Sink s = null;

            lock (m_lock_object)
            {
                m_sinks.TryGetValue(c, out s);
            }

            return s;
        }


        /// <summary>
        /// Sets a level, below which messages are filtered
        /// </summary>
        /// <param name="c">the category to filter</param>
        /// <param name="level">the level to display</param>
        public void SetLevel(UserMessage.Category c, int level)
        {
            lock (m_lock_object)
            {
                m_level[c] = level;
            }
        }

        /// <summary>
        /// Given the category, this method returns the current
        /// level for this type of message
        /// </summary>
        /// <param name="c">the type of message</param>
        /// <returns>the current level</returns>
        public int GetLevel(UserMessage.Category c)
        {
            int ret = 0;

            lock (m_lock_object)
            {
                if (m_level.ContainsKey(c))
                    ret = m_level[c];
            }

            return ret;
        }


        /// <summary>
        /// Remove the sink associated with a specific category of message.
        /// </summary>
        /// <param name="c">the category of message</param>
        public void RemoveSink(UserMessage.Category c)
        {
            lock (m_lock_object)
            {
                m_sinks.Remove(c);
            }
        }

        /// <summary>
        /// This function returns TRUE if the logger has a message sink for the given
        /// category of message.  Otherwise it returns FALSE.
        /// </summary>
        /// <param name="c">the category of interest</param>
        /// <returns>true if a sink is present, false otherwise</returns>
        public bool HasSink(UserMessage.Category c)
        {
            return m_sinks.Keys.Contains(c);
        }

        /// <summary>
        /// Log a message
        /// </summary>
        /// <param name="m">the message to log</param>
        public void LogMessage(UserMessage m)
        {
            lock (m_lock_object)
            {
                if (m_sinks.ContainsKey(m.MType))
                {
                    int level = 0;

                    if (m_level.ContainsKey(m.MType))
                        level = m_level[m.MType];

                    if (m.Level <= level)
                    {
                        Sink s = m_sinks[m.MType];
                        s.LogMessage(m);
                    }
                }
            }
        }

        /// <summary>
        /// Dump a data buffer to the log file, used for debugging
        /// </summary>
        /// <param name="loglevel">the log level</param>
        /// <param name="title">the title</param>
        /// <param name="buffer">the buffer</param>
        /// <param name="size">the buffer size</param>
        public void DumpBuffer(uint loglevel, string title, IntPtr buffer, int size)
        {
            UserMessage m;

            m = new UserMessage(UserMessage.Category.Debug, loglevel, "Dumping data for structure '" + title + "'");
            LogMessage(m);

            int index = 0;
            int linebytes = 0;
            string str = string.Empty;
            while (index < size)
            {
                if (linebytes == 16)
                {
                    m = new UserMessage(UserMessage.Category.Debug, loglevel, str);
                    LogMessage(m);
                    linebytes = 0;
                    str = string.Empty;
                }

                byte b = Marshal.ReadByte(buffer, index++);
                str += b.ToString("X2") + " ";
                linebytes++;
            }

            if (linebytes != 0)
            {
                m = new UserMessage(UserMessage.Category.Debug, loglevel, str);
                LogMessage(m);
            }
        }
    }
}
