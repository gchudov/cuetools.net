using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;
using System.Threading;
using MetadataPlugIn;
using System.Drawing.Drawing2D;

namespace CUETools.CTDB.EACPlugin
{
    public partial class ImagePreview : Panel
    {
        const int ImageSize = 80;
        protected InternetImage m_image = null;
        protected bool m_selected;
        protected int m_random;

        public ImagePreview(Control parent, MouseEventHandler parentmouse, MouseEventHandler parentclick, MouseEventHandler parentdoubleclick, InternetImage img)
        {
            InitializeComponent();
            m_random = new Random().Next(256);

            //this.BackColor = System.Drawing.Color.Black;
            //this.Location = new System.Drawing.Point(positionx*(128+8)+4, positiony*(128+16+4+8)+4);
            //this.Bounds = new Rectangle(new System.Drawing.Point(positionx * (128 + 8) + 4, positiony * (128 + 16 + 4 + 8) + 4), Size);
            this.MouseMove += new MouseEventHandler(parentmouse);
            this.MouseClick += new MouseEventHandler(parentclick);
            this.MouseDoubleClick += new MouseEventHandler(parentdoubleclick);
            this.MouseOverPanel.MouseMove += new System.Windows.Forms.MouseEventHandler(parentmouse);
            this.ImagePanel.MouseMove += new System.Windows.Forms.MouseEventHandler(parentmouse);
            this.ImagePanel.MouseClick += new MouseEventHandler(parentclick);
            this.ImagePanel.MouseDoubleClick += new MouseEventHandler(parentdoubleclick);
            this.Description.MouseClick +=new MouseEventHandler(parentclick);
            this.Description.MouseDoubleClick += new MouseEventHandler(parentdoubleclick);
            this.Description2.MouseClick += new MouseEventHandler(parentclick);
            this.Description2.MouseDoubleClick += new MouseEventHandler(parentdoubleclick);
            this.SaveFile.Click += new EventHandler(SaveFile_Click);
            this.DoubleBuffered = true;

            Image = img;
            Selected = false;
            
            parent.Controls.Add(this);
        }

        public void SaveFile_Click(object sender, EventArgs e)
        {
            try
            {

                SaveFileDialog fd = new SaveFileDialog();
                string[] URL = m_image.URL.Split('/');
                if (URL.Length > 0)
                {
                    string filename = URL[URL.Length - 1];
                    filename = filename.Replace('\\', '-');
                    filename = filename.Replace('\"', '\'');
                    filename = filename.Replace('*', '.');
                    filename = filename.Replace(':', '.');
                    filename = filename.Replace('<', '(');
                    filename = filename.Replace('>', ')');
                    filename = filename.Replace('|', 'I');
                    filename = filename.Replace('/', '-');
                    filename = filename.Replace('?', ' ');
                    fd.FileName = filename;
                    fd.Title = "Save Image File";
                    var DialogResult = fd.ShowDialog();
                    if(DialogResult == DialogResult.OK)
                    {
                        FileStream fs = File.Create(fd.FileName);
                        BinaryWriter bw = new BinaryWriter(fs);
                        bw.Write(m_image.Data);
                        bw.Close();
                        fs.Close();
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.Message);
            }
        }


        public bool Selected
        {
            get
            {
                return m_selected;
            }
            set
            {
                m_selected = value;
                if (m_selected)
                {
                    BackColor = System.Drawing.SystemColors.MenuHighlight;
                }
                else
                {
                    BackColor = System.Drawing.Color.Transparent;
                }
            }
        }


        public bool IsMouseOverPanel(Point pnt)
        {
            Point p2 = PointToClient(pnt);
            return MouseOverPanel.Bounds.Contains(p2);
        }


        public InternetImage Image
        {
            get
            {
                return m_image;
            }
            set
            {
                m_image = value;

                var imgToResize = m_image.Image;
                int sourceWidth = imgToResize.Width;
                int sourceHeight = imgToResize.Height;

                float nPercent = 0;
                float nPercentW = 0;
                float nPercentH = 0;

                nPercentW = ((float)this.ImagePanel.Width / (float)sourceWidth);
                nPercentH = ((float)this.ImagePanel.Height / (float)sourceHeight);

                if (nPercentH < nPercentW)
                    nPercent = nPercentH;
                else
                    nPercent = nPercentW;

                int destWidth = (int)(sourceWidth * nPercent);
                int destHeight = (int)(sourceHeight * nPercent);

                try
                {
                    var b = new Bitmap(this.ImagePanel.Width, this.ImagePanel.Height);
                    using (Graphics g = Graphics.FromImage((Image)b))
                    {
                        g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                        g.DrawImage(imgToResize, 0, 0, destWidth, destHeight);
                    }
                    this.ImagePanel.BackgroundImage = b;
                }
                catch
                {
                }

                this.Description.Text = m_image.Image.Width + "x" + m_image.Image.Height;
                this.Description2.Text = (m_image.Data.Length / 1024) + " kb";
            }
        }

        public string URL
        {
            get
            {
                return m_image.URL;
            }
        }

        public long FileSize
        {
            get
            {
                return ((long)m_image.Data.Length) * 256 + ((long)m_random);
            }
        }


        public long PictureSize
        {
            get
            {
                return ((long)m_image.Image.Width) * ((long)m_image.Image.Height) * 256 + ((long)m_random);
            }
        }


        private void ImagePreview_MouseMove(object sender, MouseEventArgs e)
        {
            Parent.PointToClient(PointToScreen(e.Location));
        }
    }
}


