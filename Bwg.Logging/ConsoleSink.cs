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
    /// A message sink class that writes messages to the console
    /// </summary>
    public class ConsoleSink : Sink
    {
        /// <summary>
        /// Log the message to the console
        /// </summary>
        /// <param name="m">the message</param>
        public override void LogMessage(UserMessage m)
        {
            if (m.Code == 0)
                System.Console.WriteLine(m.MType.ToString() + ": " + m.Text);
            else
                System.Console.WriteLine(m.MType.ToString() + " " + m.Code.ToString() + ": " + m.Text);
        }
    }
}
