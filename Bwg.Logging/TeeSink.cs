using System;
using System.Collections.Generic;
using System.Text;

namespace Bwg.Logging
{
    /// <summary>
    /// This class is a message sink that forwards a message to a set of receiving sinks
    /// </summary>
    public class TeeSink : Sink
    {
        #region private member variables
        /// <summary>
        /// The list of sinks to send the message to
        /// </summary>
        IList<Sink> m_sinks;
        #endregion

        #region constructor
        /// <summary>
        /// The constructor to creat the object
        /// </summary>
        public TeeSink()
        {
            m_sinks = new List<Sink>();
        }
        #endregion

        #region public methods
        /// <summary>
        /// Add the sink to the list of receiving sinks
        /// </summary>
        /// <param name="s">the sink to add</param>
        public void AddSink(Sink s)
        {
            m_sinks.Add(s);
        }

        /// <summary>
        /// Remove the sink to the list of receiving sinks
        /// </summary>
        /// <param name="s"></param>
        public void RemoveSink(Sink s)
        {
            m_sinks.Remove(s);
        }

        /// <summary>
        /// Log a message to the receiving sinks
        /// </summary>
        /// <param name="m">the message</param>
        public override void LogMessage(UserMessage m)
        {
            foreach (Sink s in m_sinks)
                s.LogMessage(m);
        }
        #endregion
    }
}
