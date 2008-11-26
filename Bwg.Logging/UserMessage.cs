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
    /// A message to be logged to the u
    /// </summary>
    public class UserMessage
    {
        #region Public Types
        /// <summary>
        /// Message types
        /// </summary>
        public enum Category
        {
            /// <summary>
            /// An error occurred
            /// </summary>
            Error,

            /// <summary>
            /// A warning occurred
            /// </summary>
            Warning,

            /// <summary>
            /// Information about the process
            /// </summary>
            Info,

            /// <summary>
            /// Debugging information
            /// </summary>
            Debug
        } ;
        #endregion

        #region Public Data Members
        /// <summary>
        /// The category for the message (error, warning, info, debug)
        /// </summary>
        public readonly Category MType ;

        /// <summary>
        /// The numeric error code
        /// </summary>
        public readonly uint Code ;

        /// <summary>
        /// The text for the error message
        /// </summary>
        public readonly string Text;

        /// <summary>
        /// The level for the message, used to filter out messages
        /// </summary>
        public readonly uint Level;

        /// <summary>
        /// This member contains a time stamp for the message
        /// </summary>
        public readonly DateTime When;
        #endregion

        #region constructor
        /// <summary>
        /// Constructor for a message
        /// </summary>
        /// <param name="t">Category of the message</param>
        /// <param name="level">Level of the message</param>
        /// <param name="s">Text of the message</param>
        public UserMessage(Category t, uint level, string s)
        {
            MType = t;
            Text = s;
            Level = level;
            When = DateTime.Now;
        }
        #endregion

        #region public member functions
        /// <summary>
        /// This method converts a user message to a single string
        /// </summary>
        /// <returns>the string that represnts the message</returns>
        public override string ToString()
        {
            return When.ToLongDateString() + " " + When.ToLongTimeString() + " : " + MType.ToString() + " : Level " + Level.ToString() + " : " + Text;
        }
        #endregion
    }
}
