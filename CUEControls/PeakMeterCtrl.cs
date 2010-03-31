///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2008 Ernest Laurentin (elaurentin@netzero.net)
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
// claim that you wrote the original software. If you use this software
// in a product, an acknowledgment in the product documentation would be
// appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not be
// misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.
///////////////////////////////////////////////////////////////////////////////
using System;
using System.ComponentModel;
using System.Drawing;
using System.Windows.Forms;
using System.Threading;

namespace Ernzo.WinForms.Controls
{
    public enum PeakMeterStyle
    {
        PMS_Horizontal = 0,
        PMS_Vertical   = 1
    }

    internal struct PeakMeterData
    {
        public int Value;
        public int Falloff;
        public int Speed;
    }

    [ToolboxBitmap(typeof(MyResourceNamespace), "pmicon.bmp")]
    public partial class PeakMeterCtrl : Control
    {
        private const byte DarkenByDefault = 40;
        private const byte LightenByDefault = 150;
        private const int MinRangeDefault = 60;
        private const int MedRangeDefault = 80;
        private const int MaxRangeDefault = 100;
        private const int FalloffFast = 1;
        private const int FalloffNormal = 10;
        private const int FalloffSlow = 100;
        private const int FalloffDefault = 10;
        private const int DecrementPercent = 10;
        private const int BandsMin = 1;
        private const int BandsMax = 1000;
        private const int BandsDefault = 8;
        private const int LEDMin  = 1;
        private const int LEDMax  = 1000;
        private const int LEDDefault = 8;
        private const int cxyMargin = 1;

        private int _AnimDelay;
        private int _MinRangeValue;
        private int _MedRangeValue;
        private int _MaxRangeValue;
        private PeakMeterData[] _meterData;
        private System.Threading.Timer _animationTimer;
        public PeakMeterCtrl()
        {
            InitializeComponent();
            InitDefault();
            this.SetStyle(ControlStyles.OptimizedDoubleBuffer, true);
            this.SetStyle(ControlStyles.UserPaint, true);
            //this.SetStyle(ControlStyles.AllPaintingInWmPaint, true);
        }

        private void InitDefault()
        {
            _AnimDelay = Timeout.Infinite;
            _MinRangeValue = MinRangeDefault; // [0,60[
            _MedRangeValue = MedRangeDefault; // [60,80[
            _MaxRangeValue = MaxRangeDefault; // [80,100[
            _meterData = null;
            _animationTimer = null;
            _ShowGrid = true;
            _ColoredGrid = false;
            _GridColor = Color.Gainsboro;
            _ColorNormal = Color.Green;
            _ColorMedium = Color.Yellow;
            _ColorHigh = Color.Red;
            _ColorNormalBack = LightenColor(_ColorNormal, LightenByDefault);
            _ColorMediumBack = LightenColor(_ColorMedium, LightenByDefault);
            _ColorHighBack = LightenColor(_ColorHigh, LightenByDefault);
            _BandsCount = BandsDefault;
            _LEDCount = LEDDefault;
            _FalloffSpeed = FalloffNormal;
            _FalloffEffect = true;
            _FalloffColor = DarkenColor(_GridColor, DarkenByDefault);
            ResetControl();
        }

        #region Control properties
        private PeakMeterStyle _PmsMeterStyle;
        [Category("Appearance"), DefaultValue(PeakMeterStyle.PMS_Horizontal)]
        public PeakMeterStyle MeterStyle
        {
            get { return _PmsMeterStyle; }
            set { _PmsMeterStyle = value; Refresh(); }
        }

        private bool _ShowGrid;
        [Category("Appearance"), DefaultValue(true)]
        public bool ShowGrid
        {
            get { return _ShowGrid; }
            set { _ShowGrid = value; Refresh(); }
        }

        private bool _ColoredGrid;
        [Category("Appearance"), DefaultValue(false)]
        public bool ColoredGrid
        {
            get { return _ColoredGrid; }
            set { _ColoredGrid = value; Refresh(); }
        }

        private Color _GridColor;
        [Category("Appearance")]
        public Color GridColor
        {
            get { return _GridColor; }
            set { _GridColor = value; Refresh(); }
        }

        private Color _ColorNormal;
        [Category("Appearance")]
        public Color ColorNormal
        {
            get { return _ColorNormal; }
            set { _ColorNormal = value; Refresh(); }
        }

        private Color _ColorMedium;
        [Category("Appearance")]
        public Color ColorMedium
        {
            get { return _ColorMedium; }
            set { _ColorMedium = value; Refresh(); }
        }

        private Color _ColorHigh;
        [Category("Appearance")]
        public Color ColorHigh
        {
            get { return _ColorHigh; }
            set { _ColorHigh = value;  Refresh(); }
        }

        private Color _ColorNormalBack;
        [Category("Appearance")]
        public Color ColorNormalBack
        {
            get { return _ColorNormalBack; }
            set { _ColorNormalBack = value; Refresh(); }
        }

        private Color _ColorMediumBack;
        [Category("Appearance")]
        public Color ColorMediumBack
        {
            get { return _ColorMediumBack; }
            set { _ColorMediumBack = value; Refresh(); }
        }

        private Color _ColorHighBack;
        [Category("Appearance")]
        public Color ColorHighBack
        {
            get { return _ColorHighBack; }
            set { _ColorHighBack = value; Refresh(); }
        }

        private int _BandsCount;
        [Category("Appearance"), DefaultValue(BandsDefault)]
        public int BandsCount
        {
            get { return _BandsCount; }
            set {
                if (value >= BandsMin && value <= BandsMax)
                {
                    _BandsCount = value;
                    ResetControl();
                    Refresh();
                }
            }
        }

        private int _LEDCount;
        [Category("Appearance"), DefaultValue(LEDDefault)]
        public int LEDCount
        {
            get { return _LEDCount; }
            set {
                if (value >= LEDMin && value <= LEDMax)
                {
                    _LEDCount = value;
                    Refresh();
                }
            }
        }

        private int _FalloffSpeed;
        [Category("Falloff Effect"), DefaultValue(FalloffDefault)]
        public int FalloffSpeed
        {
            get { return _FalloffSpeed; }
            set { _FalloffSpeed = value; }
        }

        private bool _FalloffEffect;
        [Category("Falloff Effect"), DefaultValue(true)]
        public bool FalloffEffect
        {
            get { return _FalloffEffect; }
            set { _FalloffEffect = value; }
        }

        private Color _FalloffColor;
        [Category("Falloff Effect")]
        public Color FalloffColor
        {
            get { return _FalloffColor; }
            set { _FalloffColor = value; }
        }
	
        #endregion

        [Browsable(false)]
        public bool IsActive
        {
            get { return (_animationTimer != null); }
        }

        #region Control Methods

        // support for thread-safe version
        private delegate bool StartDelegate(int delay);
        /// <summary>
        /// Start animation
        /// </summary>
        /// <param name="delay"></param>
        /// <returns></returns>
        public bool Start(int delay)
        {
            if (this.InvokeRequired)
            {
                StartDelegate startDelegate = new StartDelegate(this.Start);
                return (bool)this.Invoke(startDelegate, delay);
            }
            _AnimDelay = delay;
            return StartAnimation(delay);
        }

        // support for thread-safe version
        private delegate bool StopDelegate();
        /// <summary>
        /// Stop Animation
        /// </summary>
        /// <returns></returns>
        public bool Stop()
        {
            if (this.InvokeRequired)
            {
                StopDelegate stopDelegate = new StopDelegate(this.Stop);
                return (bool)this.Invoke(stopDelegate);
            }
            _AnimDelay = Timeout.Infinite;
            return StopAnimation();
        }

        /// <summary>
        /// Set number of LED bands
        /// </summary>
        /// <param name="BandsCount">Number of bands</param>
        /// <param name="LEDCount">Number of LED per bands</param>
        public void SetMeterBands(int BandsCount, int LEDCount)
        {
            if (BandsCount < BandsMin || BandsCount > BandsMax)
                throw new ArgumentOutOfRangeException("BandsCount");
            if (LEDCount < LEDMin || LEDCount > LEDMax)
                throw new ArgumentOutOfRangeException("LEDCount");
            _BandsCount = BandsCount;
            _LEDCount = LEDCount;
            ResetControl();
            Refresh();
        }

        /// <summary>
        /// Set range info
        /// </summary>
        /// <param name="minRangeVal">Min Range</param>
        /// <param name="medRangeVal">Medium Range</param>
        /// <param name="maxRangeVal">High Range</param>
        public void SetRange(int minRangeVal, int medRangeVal, int maxRangeVal)
        {
        	if (maxRangeVal <= medRangeVal || medRangeVal < minRangeVal )
                throw new ArgumentOutOfRangeException("minRangeVal");
            _MinRangeValue = minRangeVal;
            _MedRangeValue = medRangeVal;
            _MaxRangeValue = maxRangeVal;
            ResetControl();
            Refresh();
        }

        // support for thread-safe version
        private delegate bool SetDataDelegate(int[] arrayValue, int offset, int size);
        /// <summary>
        /// Set meter band value
        /// </summary>
        /// <param name="arrayValue">Array value for the bands</param>
        /// <param name="offset">Starting offset position</param>
        /// <param name="size">Number of values to set</param>
        /// <returns></returns>
        public bool SetData(int[] arrayValue, int offset, int size)
        {
            if (arrayValue == null)
                throw new ArgumentNullException("arrayValue");
            if (arrayValue.Length < (offset + size))
                throw new ArgumentOutOfRangeException("arrayValue");

            if (this.InvokeRequired)
            {
                SetDataDelegate setDelegate = new SetDataDelegate(this.SetData);
                return (bool)this.Invoke(setDelegate, arrayValue, offset, size);
            }
            bool isRunning = this.IsActive;

            Monitor.Enter(this._meterData);

            int maxIndex = offset + size;
            for (int i = offset; i < maxIndex; i++)
            {
                if (i < this._meterData.Length)
                {
                    PeakMeterData pm = this._meterData[i];
                    pm.Value = Math.Min(arrayValue[i], this._MaxRangeValue);
                    pm.Value = Math.Max(pm.Value, 0);
                    if (pm.Falloff < pm.Value)
                    {
                        pm.Falloff = pm.Value;
                        pm.Speed = this._FalloffSpeed;
                    }
                    this._meterData[i] = pm;
                }
            }
            Monitor.Exit(this._meterData);
 
            // check that timer should be restarted
            if (_AnimDelay != Timeout.Infinite)
            {
                if (_animationTimer == null)
                {
                    StartAnimation(_AnimDelay);
                }
            }
            else
            {
                Refresh();
            }

            return isRunning;
        }
        #endregion

        /// <summary>
        /// Make a color darker
        /// </summary>
        /// <param name="color">Color to darken</param>
        /// <param name="darkenBy">Value to decrease by</param>
        protected virtual Color DarkenColor(Color color, byte darkenBy)
        {
            byte red = (byte)(color.R > darkenBy ? (color.R - darkenBy) : 0);
            byte green = (byte)(color.G > darkenBy ? (color.G - darkenBy) : 0);
            byte blue = (byte)(color.B > darkenBy ? (color.B - darkenBy) : 0);
            return Color.FromArgb(red, green, blue);
        }
        /// <summary>
        /// Make a color lighter
        /// </summary>
        /// <param name="color"></param>
        /// <param name="lightenBy"></param>
        /// <returns></returns>
        protected virtual Color LightenColor(Color color, byte lightenBy)
        {
            byte red = (byte)((color.R + lightenBy) <= 255 ? (color.R + lightenBy) : 255);
            byte green = (byte)((color.G + lightenBy) <= 255 ? (color.G + lightenBy) : 255);
            byte blue = (byte)((color.B + lightenBy) <= 255 ? (color.B + lightenBy) : 255);
            return Color.FromArgb(red, green, blue);
        }

        protected static bool InRange(int value, int rangeMin, int rangeMax)
        {
            return (value >= rangeMin && value <= rangeMax);
        }
        
        protected void ResetControl()
        {
            _meterData = new PeakMeterData[_BandsCount];
            PeakMeterData pm;
            pm.Value = _MaxRangeValue;
            pm.Falloff = _MaxRangeValue;
            pm.Speed = _FalloffSpeed;
            for (int i = 0; i < _meterData.Length; i++)
            {
                _meterData[i] = pm;
            }
        }
        protected bool StartAnimation(int period)
        {
            if ( !IsActive )
            {
                TimerCallback timerDelegate = 
                    new TimerCallback(TimerCallback);
                _animationTimer = new System.Threading.Timer(timerDelegate, this, Timeout.Infinite, Timeout.Infinite);
            }
            return _animationTimer.Change(period, period);
        }
        protected bool StopAnimation()
        {
            bool result = false;
            if ( IsActive )
            {
                try
                {
                    result = _animationTimer.Change(Timeout.Infinite, Timeout.Infinite);
                    _animationTimer.Dispose();
                    _animationTimer = null;
                    result = true;
                }
                catch (Exception)
                {
                }
            }
            return result;
        }

        protected override void OnHandleDestroyed(EventArgs e)
        {
            Stop();
            base.OnHandleDestroyed(e);
        }
        protected override void OnBackColorChanged(EventArgs e)
        {
            Refresh();
        }
        protected override void OnPaint(PaintEventArgs e)
        {
            // Calling the base class OnPaint
            base.OnPaint(e);

            Graphics g = e.Graphics;
            Rectangle rect = new Rectangle(0, 0, this.Width, this.Height);
            Brush backColorBrush = new SolidBrush(this.BackColor);

            g.FillRectangle(backColorBrush, rect);
            //rect.Inflate(-this.Margin.Left, -this.Margin.Top);
            if (MeterStyle == PeakMeterStyle.PMS_Horizontal)
            {
                DrawHorzBand(g, rect);
            }
            else
            {
                DrawVertBand(g, rect);
            }
        }

        protected void TimerCallback(Object thisObject)
        {
            try
            {
                // refresh now!
                Control thisControl = thisObject as Control;
                if (thisControl != null && thisControl.IsHandleCreated)
                {
                    thisControl.Invoke(new MethodInvoker(Refresh));
                }
                else
                {
                    return;
                }
            }
            catch (Exception)
            {
                // just ignore
            }

            int nDecValue  = _MaxRangeValue / _LEDCount;
            bool noUpdate = true;

            Monitor.Enter(this._meterData);
            for (int i = 0; i < _meterData.Length; i++)
            {
                PeakMeterData pm = _meterData[i];

                if (pm.Value > 0)
                {
                    pm.Value -= (_LEDCount > 1 ? nDecValue : (_MaxRangeValue * DecrementPercent) / 100);
                    if (pm.Value < 0)
                        pm.Value = 0;
                    noUpdate = false;
                }

                if (pm.Speed > 0)
                {
                    pm.Speed -= 1;
                    noUpdate = false;
                }

                if (pm.Speed == 0 && pm.Falloff > 0)
                {
                    pm.Falloff -= (_LEDCount > 1 ? nDecValue >> 1 : 5);
                    if (pm.Falloff < 0)
                        pm.Falloff = 0;
                    noUpdate = false;
                }

                // re-assign PeakMeterData
                _meterData[i] = pm;
            }
            Monitor.Exit(this._meterData);

            if (noUpdate) // Stop timer if no more data but do not reset ID
            {
                StopAnimation();
            }
        }
        protected void DrawHorzBand(Graphics g, Rectangle rect)
        {
            int nMaxRange = (_MedRangeValue == 0) ? Math.Abs(_MaxRangeValue - _MinRangeValue) : _MaxRangeValue;
            int nVertBands = (_LEDCount > 1 ? _LEDCount : (nMaxRange * DecrementPercent) / 100);
            int nMinVertLimit = _MinRangeValue * nVertBands / nMaxRange;
            int nMedVertLimit = _MedRangeValue * nVertBands / nMaxRange;
            int nMaxVertLimit = nVertBands;

            if (_MedRangeValue == 0)
            {
                nMedVertLimit = Math.Abs(_MinRangeValue) * nVertBands / nMaxRange;
                nMinVertLimit = 0;
            }
            Size size = new Size(rect.Width/_BandsCount, rect.Height/nVertBands);
            Rectangle rcBand = new Rectangle(rect.Location, size);

            // Draw band from bottom
            rcBand.Offset(0, rect.Height-size.Height);
            int xDecal = (_BandsCount>1 ? cxyMargin : 0);
            //int yDecal = 0;

            Color gridPenColor = (this.ShowGrid ? GridColor : BackColor);
            Pen gridPen = new Pen(gridPenColor);
            Pen fallPen = new Pen( this.FalloffColor );

            for(int nHorz=0; nHorz < _BandsCount; nHorz++)
            {
                int nValue = _meterData[nHorz].Value;
                int nVertValue = nValue*nVertBands/nMaxRange;
                Rectangle rcPrev = rcBand;

                for(int nVert=0; nVert < nVertBands; nVert++)
                {
                    // Find color based on range value
                    Color colorBand = gridPenColor;

                    // Draw grid line (level) bar
                    if ( this.ShowGrid && (nVert == nMinVertLimit || nVert == nMedVertLimit || nVert == (nVertBands-1)))
                    {
                        Point []points = new Point[2];
                        points[0].X = rcBand.Left;
                        points[0].Y = rcBand.Top + (rcBand.Height>>1);
                        points[1].X = rcBand.Right;
                        points[1].Y = points[0].Y;
                        g.DrawPolygon(gridPen, points);
                    }

                    if ( _MedRangeValue == 0 )
                    {
                        int nVertStart = nMedVertLimit+nVertValue;
                        if ( InRange(nVert, nVertStart, nMedVertLimit-1) )
                            colorBand = this.ColorNormal;
                        else if ( nVert >= nMedVertLimit && InRange(nVert, nMedVertLimit, nVertStart) )
                            colorBand = this.ColorHigh;
                        else {
                            colorBand = (nVert < nMedVertLimit) ? this.ColorNormalBack : this.ColorHighBack;
                        }
                    }
                    else if ( nVertValue < nVert )
                    {
                        if ( this.ShowGrid && this.ColoredGrid )
                        {
                            if ( InRange(nVert, 0, nMinVertLimit) )
                                colorBand = this.ColorNormalBack;
                            else if ( InRange(nVert, nMinVertLimit+1, nMedVertLimit) )
                                colorBand = this.ColorMediumBack;
                            else if ( InRange(nVert, nMedVertLimit+1, nMaxVertLimit) )
                                colorBand = this.ColorHighBack;
                        }
                    } else {
                        if (nValue == 0)
                        {
                            if (this.ShowGrid && this.ColoredGrid)
                                colorBand = this.ColorNormalBack;
                        }
                        else if ( InRange(nVert, 0, nMinVertLimit) )
                            colorBand = this.ColorNormal;
                        else if ( InRange(nVert, nMinVertLimit+1, nMedVertLimit) )
                            colorBand = this.ColorMedium;
                        else if ( InRange(nVert, nMedVertLimit+1, nMaxVertLimit) )
                            colorBand = this.ColorHigh;
                    }

                    if (colorBand != this.BackColor)
                    {
                        SolidBrush fillBrush = new SolidBrush(colorBand);
                        if (this._LEDCount > 1)
                            rcBand.Inflate(-cxyMargin, -cxyMargin);
                        g.FillRectangle(fillBrush, rcBand.Left, rcBand.Top, rcBand.Width+1, rcBand.Height);
                        if (this._LEDCount > 1)
                            rcBand.Inflate(cxyMargin, cxyMargin);
                    }
                    rcBand.Offset(0, -size.Height);
                }

                // Draw falloff effect
                if (this.FalloffEffect && this.IsActive)
                {
                    int nMaxHeight = size.Height*nVertBands;
                    Point []points = new Point[2];
                    points[0].X = rcPrev.Left + xDecal;
                    points[0].Y = rcPrev.Bottom - (_meterData[nHorz].Falloff * nMaxHeight) / _MaxRangeValue;
                    points[1].X = rcPrev.Right - xDecal;
                    points[1].Y = points[0].Y;
                    g.DrawPolygon(fallPen, points);
                }

                // Move to Next Horizontal band
                rcBand.Offset(size.Width, size.Height * nVertBands);
            }
        }
        protected void DrawVertBand(Graphics g, Rectangle rect)
        {
            int nMaxRange = (_MedRangeValue == 0) ? Math.Abs(_MaxRangeValue - _MinRangeValue) : _MaxRangeValue;
            int nHorzBands = (_LEDCount > 1 ? _LEDCount : (nMaxRange * DecrementPercent) / 100);
	        int nMinHorzLimit = _MinRangeValue*nHorzBands/nMaxRange;
	        int nMedHorzLimit = _MedRangeValue*nHorzBands/nMaxRange;
	        int nMaxHorzLimit = nHorzBands;

	        if ( _MedRangeValue == 0 )
	        {
		        nMedHorzLimit = Math.Abs(_MinRangeValue)*nHorzBands/nMaxRange;
		        nMinHorzLimit = 0;
	        }

            Size size = new Size(rect.Width/nHorzBands, rect.Height/_BandsCount);
            Rectangle rcBand = new Rectangle(rect.Location, size);

	        // Draw band from top
            rcBand.Offset(0, rect.Height-size.Height*_BandsCount);
	        //int xDecal = 0;
	        int yDecal = (_BandsCount>1 ? cxyMargin : 0);

            Color gridPenColor = (this.ShowGrid ? GridColor : BackColor);
            Pen gridPen = new Pen(gridPenColor);
            Pen fallPen = new Pen( this.FalloffColor );

            for(int nVert=0; nVert < _BandsCount; nVert++)
	        {
                int nValue = _meterData[nVert].Value;
		        int nHorzValue = nValue*nHorzBands/nMaxRange;
                Rectangle rcPrev = rcBand;

		        for(int nHorz=0; nHorz < nHorzBands; nHorz++)
		        {
                    // Find color based on range value
                    Color colorBand = gridPenColor;

			        if ( this.ShowGrid && (nHorz == nMinHorzLimit || nHorz == nMedHorzLimit || nHorz == (nHorzBands-1)))
			        {
                        Point []points = new Point[2];
				        points[0].X = rcBand.Left + (rcBand.Width>>1);
				        points[0].Y = rcBand.Top;
				        points[1].X = points[0].X;
				        points[1].Y = rcBand.Bottom;
                        g.DrawPolygon(gridPen, points);
                    }

                    if (_MedRangeValue == 0)
			        {
				        int nHorzStart = nMedHorzLimit+nHorzValue;
				        if ( InRange(nHorz, nHorzStart, nMedHorzLimit-1) )
					        colorBand = this.ColorNormal;
				        else if ( nHorz >= nMedHorzLimit && InRange(nHorz, nMedHorzLimit, nHorzStart) )
					        colorBand = this.ColorHigh;
				        else {
					        colorBand = (nHorz < nMedHorzLimit) ? this.ColorNormalBack : this.ColorHighBack;
				        }
			        }
			        else if ( nHorzValue < nHorz )
			        {
				        if ( this.ShowGrid && this.ColoredGrid )
				        {
					        if ( InRange(nHorz, 0, nMinHorzLimit) )
						        colorBand = this.ColorNormalBack;
					        else if ( InRange(nHorz, nMinHorzLimit+1, nMedHorzLimit) )
						        colorBand = this.ColorMediumBack;
					        else if ( InRange(nHorz, nMedHorzLimit+1, nMaxHorzLimit) )
						        colorBand = this.ColorHighBack;
				        }
			        } else {
                        if (nValue == 0)
                        {
                            if (this.ShowGrid && this.ColoredGrid)
                                colorBand = this.ColorNormalBack;
                        }
                        else if (InRange(nHorz, 0, nMinHorzLimit))
					        colorBand = this.ColorNormal;
				        else if ( InRange(nHorz, nMinHorzLimit+1, nMedHorzLimit) )
					        colorBand = this.ColorMedium;
				        else if ( InRange(nHorz, nMedHorzLimit+1, nMaxHorzLimit) )
					        colorBand = this.ColorHigh;
			        }

                    if (colorBand != this.BackColor)
                    {
                        SolidBrush fillBrush = new SolidBrush(colorBand);
                        if (this._LEDCount > 1)
                            rcBand.Inflate(-cxyMargin, -cxyMargin);
                        g.FillRectangle(fillBrush, rcBand.Left, rcBand.Top, rcBand.Width, rcBand.Height+1);
                        if (this._LEDCount > 1)
                            rcBand.Inflate(cxyMargin, cxyMargin);
                    }
                    rcBand.Offset(size.Width, 0);
		        }

                // Draw falloff effect
                if (this.FalloffEffect && this.IsActive)
                {
                    int nMaxWidth = size.Width * nHorzBands;
                    Point[] points = new Point[2];
                    points[0].X = rcPrev.Left + (_meterData[nVert].Falloff * nMaxWidth) / _MaxRangeValue;
                    points[0].Y = rcPrev.Top + yDecal;
                    points[1].X = points[0].X;
                    points[1].Y = rcPrev.Bottom - yDecal;
                    g.DrawPolygon(fallPen, points);
                }
                
                // Move to Next Vertical band
		        rcBand.Offset(-size.Width*nHorzBands, size.Height);
	        }
        }
    }
}

// use this to find resource namespace
internal class MyResourceNamespace
{
}
