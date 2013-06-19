using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs
{
    /// <summary>
    /// Default property value for each encoder mode attribute
    /// </summary>
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public class DefaultValueForModeAttribute : Attribute
    {
        /// <summary>
        /// Resource manager to use;
        /// </summary>
        public int[] m_values;

        /// <summary>
        /// Construct the description attribute
        /// </summary>
        /// <param name="text"></param>
        public DefaultValueForModeAttribute(params int[] values)
        {
            this.m_values = values;
        }
    }
}
