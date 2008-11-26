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

namespace Bwg.Logging
{
    /// <summary>
    /// This class is a message sink that stores all of the messages
    /// in a list.
    /// </summary>
    public class MemorySink : Sink
    {
        private Object m_lock;
        private IList<UserMessage> m_list ;

        /// <summary>
        /// 
        /// </summary>
        public EventHandler<MessageAddedArgs> MessageAdded;

        /// <summary>
        /// 
        /// </summary>
        public MemorySink()
        {
            m_lock = new Object();
            m_list = new List<UserMessage>();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public int GetMessageCount()
        {
            int cnt;

            lock (m_lock)
            {
                cnt = m_list.Count;
            }
            return cnt;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public UserMessage GetMessage(int n)
        {
            UserMessage m;

            lock (m_lock)
            {
                m = m_list[n];
            }

            return m;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m"></param>
        public override void LogMessage(UserMessage m)
        {
            lock (m_lock)
            {
                m_list.Add(m);
            }

            if (MessageAdded != null)
            {
                MessageAddedArgs args = new MessageAddedArgs(m);
                MessageAdded(this, args);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public void Clear()
        {
            lock (m_lock)
            {
                m_list.Clear();
            }
        }
    }
}
