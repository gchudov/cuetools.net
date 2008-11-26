using System;
using System.Collections.Generic;
using System.Text;
using System.IO ;

namespace Bwg.Logging
{
    /// <summary>
    /// This class is a message sink that writes the message to a file
    /// </summary>
    public class FileSink : Sink, IDisposable
    {
        #region private member variables
        private TextWriter m_writer;
        #endregion

        #region constructor
        /// <summary>
        /// Create a new file sink given the name of the output file
        /// </summary>
        /// <param name="filename">the output file</param>
        public FileSink(string filename)
        {
            try
            {
                m_writer = new StreamWriter(filename);
            }
            catch (Exception)
            {
                m_writer = null;
            }
        }
        #endregion

        #region public methods
        /// <summary>
        /// Dispose of the class, close the file
        /// </summary>
        public void Dispose()
        {
            Close();
        }

        /// <summary>
        /// Close the file
        /// </summary>
        public void Close()
        {
            if (m_writer != null)
            {
                m_writer.Flush();
                m_writer.Close();
            }
        }

        /// <summary>
        /// Log the message
        /// </summary>
        /// <param name="m">the message to log</param>
        public override void LogMessage(UserMessage m)
        {
            if (m_writer != null)
                m_writer.WriteLine(m.ToString());
        }
        #endregion
    }
}
