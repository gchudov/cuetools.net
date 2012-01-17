using System;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.IO;

namespace CUETools.CTDB.EACPlugin
{
    public class InternetImage
    {
        protected string m_URL = null;
        protected Bitmap m_bitmap = null;
        protected byte[] m_data = null;

        public string URL
        {
            get
            {
                return m_URL;
            }
            set
            {
                m_URL = value;
            }
        }


        public byte[] Data
        {
            get
            {
                return m_data;
            }
            set
            {
                m_data = value;
            }
        }


        public Bitmap Image
        {
            get
            {
                return m_bitmap;
            }
            set
            {
                m_bitmap = value;
            }
        }
    }
}
