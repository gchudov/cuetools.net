#region Author/About
/************************************************************************************
    *  MediaSlider v1.3                                                                 *
    *                                                                                   *
    *  Created:     Febuary 7, 2010                                                     *
    *  Built on:    Win7                                                                *
    *  Purpose:     Animated slider control                                             *
    *  Revision:    1.3                                                                 *
    *  Tested On:   Win7 32bit, Vista 64bit, XP Professional                            *
    *  IDE:         C# 2008 SP1 FW 3.5                                                  *
    *  Referenced:  Control Library VTD                                                 *
    *  Author:      John Underhill (Steppenwolfe)                                       *
    *                                                                                   *
    *************************************************************************************

    You can not:
    -Sell or redistribute this code or the binary for profit.
    -Use this in spyware, malware, or any generally acknowledged form of malicious software.
    -Remove or alter the above author accreditation, or this disclaimer.

    You can:
    -Use this code in your applications in any way you like.
    -Use this in a published program, (a credit to vtdev.com would be nice)

    I will not:
    -Except any responsibility for this code whatsoever.
    -Modify on demand.. you have the source code, read it, learn from it, write it.
    -There is no guarantee of fitness, nor should you have any expectation of support. 
    -I further renounce any and all responsibilities for this code, in every way conceivable, 
    now, and for the rest of time.
    
    Updates to 1.1
    -fixed bug in incremental value
    -added jump to position when track clicked w/ SmoothScrolling enabled
    -fixed bug in ButtonSize property set
    -fixed a couple of things in example form
    
    Updates to 1.2
    -Fixed false error condition on Value init
    -Fixed ButtonSize property to update at design time
    -Removed native properties that are incompatible with control
    -Fixed control so that Minimum and Maximum values can both be negative -(but Maximum has to be more then Minimum)
    
    Updates to 1.3
    Fixed scrolling ceter pointer in button 'issue'
    Fixed button resize bug
    Fixed track resize bug
    Fixed backwards scroll step bug
    
    Cheers,
    John
    steppenwolfe_2000@yahoo.com
    */
#endregion

#region Directives
using System;
using System.Text;
using System.Timers;
using System.Windows.Forms;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;
using System.Diagnostics;
#endregion

namespace MediaSlider
{
    [DefaultEvent("ValueChanged"), ToolboxBitmap(typeof(MediaSlider), "tbimage")]
    public partial class MediaSlider : UserControl
    {
        #region Constants
        private const int TrackMinDepth = 2;
        private const int TrackMaxDepth = 6;
        #endregion

        #region Enums
        public enum AnimateSpeed : int
        {
            Fast = 1,
            Normal = 5,
            Slow = 20
        }

        public enum ButtonType : uint
        {
            Round = 0,
            RoundedRectInline,
            RoundedRectOverlap,
            PointerUpRight,
            PointerDownLeft,
            Hybrid,
            GlassInline,
            GlassOverlap
        }

        public enum FlyOutStyle : int
        {
            None = 0,
            OnFocus,
            Persistant
        }

        public enum PresetType : uint
        {
            WmpVolume,
            WmpTrackbar,
            WmcTrackBar,
            Office2007,
            Glass
        }

        public enum TickMode : int
        {
            Standard = 0,
            Composite,
            Precision,
            LargeStepped
        }

        public enum TrackType : uint
        {
            Progress = 0,
            Value
        }

        private enum PointDirection : int
        {
            Bottom = 0,
            Right
        }
        private enum SliderSelectedState : uint
        {
            None = 0,
            Disabled,
            Focused,
            MouseLeave,
            Pressed,
            Depressed,
            Hover
        }

        private enum ChangeType : uint
        {
            Large = 0,
            Small
        }

        private enum HitTest : uint
        {
            Nowhere = 0,
            Button,
            Track
        }
        #endregion

        #region Structs
        [StructLayout(LayoutKind.Sequential)]
        private struct RECT
        {
            public RECT(int x, int y, int right, int bottom)
            {
                this.Left = x;
                this.Top = y;
                this.Right = right;
                this.Bottom = bottom;
            }
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }
        #endregion

        #region API
        [DllImport("user32.dll")]
        private static extern IntPtr GetDC(IntPtr handle);

        [DllImport("user32.dll")]
        private static extern int ReleaseDC(IntPtr handle, IntPtr hdc);

        [DllImport("gdi32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool BitBlt(IntPtr hdc, int nXDest, int nYDest, int nWidth, int nHeight, IntPtr hdcSrc, int nXSrc, int nYSrc, int dwRop);

        [DllImport("user32.dll")]
        private static extern bool ValidateRect(IntPtr hWnd, ref RECT lpRect);

        [DllImport("user32.dll")]
        private static extern bool GetClientRect(IntPtr hWnd, ref RECT r);

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetCursorPos(ref Point lpPoint);

        [DllImport("user32.dll")]
        private static extern bool ScreenToClient(IntPtr hWnd, ref Point lpPoint);

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool PtInRect(ref RECT lprc, Point pt);
        #endregion

        #region Fields
        private bool _bHasInitialized = false;
        private bool _bPulseTimerOn = false;
        private bool _bAnimated = false;
        private bool _bSmoothScrolling = false;
        private bool _bTrackShadow = false;
        private bool _bShowButtonOnHover = false;
        private bool _bInTarget = false;
        private bool _bPropInit = false;
        private bool _bFinishedPropRun = false;
        private int _iLargeChange = 0;
        private int _iMaximum = 10;
        private int _iMinimum = 0;
        private int _iSmallChange = 0;
        private int _iValue = 0;
        private int _iTrackDepth = 6;
        private int _iButtonCornerRadius = 4;
        private int _iTrackPadding = 2;
        private int _iTickMaxLength = 4;
        private int _iTickMinPadding = 2;
        private int _iFlyOutMaxDepth = 16;
        private int _iFlyOutMaxWidth = 12;
        private int _iFlyOutSpacer = 6;
        private int _iLinePos = 0;
        private double _dButtonPosition = 0;
        private float _fAnimationSize = .1f;

        private Color _clrButtonAccentColor = Color.FromArgb(128, 64, 64, 64);
        private Color _clrButtonBorderColor = Color.Black;
        private Color _clrButtonColor = Color.FromArgb(160, 0, 0, 0);
        private Color _clrBackColor = Color.Black;
        private Color _clrTrackBorderColor = Color.FromArgb(160, Color.White);
        private Color _clrTrackFillColor = Color.Transparent;
        private Color _clrTrackProgressColor = Color.FromArgb(5, 101, 188);
        private Color _clrTickColor = Color.DarkGray;
        private Color _clrTrackShadowColor = Color.DarkGray;

        private Size _szButtonSize = new Size(14, 14);
        private Size _szMinSize = new Size(0, 0);
        private Size _szMaxSize = new Size(0, 0);
        private ButtonType _eButtonStyle = ButtonType.Round;
        private TrackType _eTrackStyle = TrackType.Value;
        private SliderSelectedState _eSliderState = SliderSelectedState.None;
        private Orientation _eOrientation = Orientation.Horizontal;
        private TickMode _eTickType = TickMode.Standard;
        private TickStyle _eTickStyle = TickStyle.None;
        private FlyOutStyle _eFlyOutOnFocus = FlyOutStyle.None;
        private AnimateSpeed _eAnimationSpeed = AnimateSpeed.Normal;

        private RECT _trackRect;
        private RECT _buttonRect;

        private Bitmap _bmpBackground;
        private Bitmap _bmpButton;
        private Bitmap _bmpSprite;
        private FadeTimer _cPulseTimer;
        private cStoreDc _cControlDc;
        private cStoreDc _cTrackDc;
        private cStoreDc _cAnimationDc;
        private ErrorProvider ErrorHandler;

        private delegate void ResetCallback();

        [Description("Callback event for the FlyOut window")]
		public event EventHandler<FlyOutEventArgs> FlyOutInfo;
        #endregion

        #region Events and Delegates
        [Description("Raised when the Slider Value property changes")]
		public event EventHandler ValueChanged;
        [Description("Raised when the mSlider has scrolled.")]
		public event EventHandler Scrolled;
        #endregion

        #region Constructor
        public MediaSlider()
        {
            Init();
            InitializeComponent();
        }

        private void Init()
        {
            CreateGraphicsObjects();
            _bHasInitialized = true;

            if (this.DesignMode)
            {
                ErrorHandler = new ErrorProvider();
                this.Minimum = 0;
                this.Maximum = 10;
                this.SmallChange = 1;
                this.LargeChange = 2;
            }
            _clrBackColor = Color.FromKnownColor(KnownColor.Control);
        }
        #endregion

        #region Destructor
        private void DeInit()
        {
            _bHasInitialized = false;
            DestroyGraphicsObjects();
            if (ErrorHandler != null)
                ErrorHandler.Dispose();
        }

        public void Dispose()
        {
            DeInit();
        }
        #endregion

        #region Properties
        #region Hidden Properties
        [Browsable(false)]
        public new bool AllowDrop
        {
            get { return base.AllowDrop; }
            set { base.AllowDrop = value; }
        }
        [Browsable(false)]
        public new AnchorStyles Anchor
        {
            get { return base.Anchor; }
            set { base.Anchor = value; }
        }
        [Browsable(false)]
        public new bool AutoScroll
        {
            get { return base.AutoScroll; }
            set { base.AutoScroll = value; }
        }
        [Browsable(false)]
        public new Size AutoScrollMargin
        {
            get { return base.AutoScrollMargin; }
            set { base.AutoScrollMargin = value; }
        }
        [Browsable(false)]
        public new Size AutoScrollMinSize
        {
            get { return base.AutoScrollMinSize; }
            set { base.AutoScrollMinSize = value; }
        }
        [Browsable(false)]
        public new AutoSizeMode AutoSizeMode
        {
            get { return base.AutoSizeMode; }
            set { base.AutoSizeMode = value; }
        }
        [Browsable(false)]
        public new AutoValidate AutoValidate
        {
            get { return base.AutoValidate; }
            set { base.AutoValidate = value; }
        }
        [Browsable(false)]
        public new ImageLayout BackgroundImageLayout
        {
            get { return base.BackgroundImageLayout; }
            set { base.BackgroundImageLayout = value; }
        }
        [Browsable(false)]
        public new ContextMenu ContextMenuStrip
        {
            get { return base.ContextMenu; }
            set { base.ContextMenu = value; }
        }
        [Browsable(false)]
        public new DockStyle Dock
        {
            get { return base.Dock; }
            set { base.Dock = value; }
        }
        [Browsable(false)]
        public new Font Font
        {
            get { return base.Font; }
            set { base.Font = value; }
        }
        [Browsable(false)]
        public new Color ForeColor
        {
            get { return base.ForeColor; }
            set { base.ForeColor = value; }
        }
        [Browsable(false)]
        public new RightToLeft RightToLeft
        {
            get { return base.RightToLeft; }
            set { base.RightToLeft = value; }
        }
        [Browsable(false)]
        public new Padding Padding
        {
            get { return base.Padding; }
            set { base.Padding = value; }
        }
        #endregion

        #region Private Properties
        private int TrackPadding
        {
            get { return _iTrackPadding; }
            set { _iTrackPadding = value; }
        }

        private bool FinishedPropRun
        {
            get { return _bFinishedPropRun; }
            set { _bFinishedPropRun = value; }
        }

        private int FlyOutMaxWidth
        {
            get { return _iFlyOutMaxWidth; }
            set { _iFlyOutMaxWidth = value; }
        }

        private int FlyOutMaxDepth
        {
            get { return _iFlyOutMaxDepth; }
            set { _iFlyOutMaxDepth = value; }
        }

        private int FlyOutSpacer
        {
            get { return _iFlyOutSpacer; }
            set { _iFlyOutSpacer = value; }
        }

        private bool InTarget
        {
            get { return _bInTarget; }
            set { _bInTarget = value; }
        }

        private int TickMaxLength
        {
            get { return _iTickMaxLength; }
            set { _iTickMaxLength = value; }
        }

        private int TickMinPadding
        {
            get { return _iTickMinPadding; }
            set { _iTickMinPadding = value; }
        }
        #endregion

        #region Public Properties
        /// <summary>Run the animation effect when focused</summary>
        [Browsable(true), Category("Appearence"),
        Description("Run the animation effect when focused")]
        public bool Animated
        {
            get { return _bAnimated; }
            set
            {
                _bAnimated = value;
                if (!this.DesignMode)
                {
                    if (this.Focused && _bAnimated)
                        StartPulseTimer();
                    else
                        StopPulseTimer();
                    DrawSlider();
                }
            }
        }

        /// <summary>Animation cycle speed</summary>
        [Browsable(true), Category("Appearence"),
        Description("Animation cycle speed")]
        public AnimateSpeed AnimationSpeed
        {
            get { return _eAnimationSpeed; }
            set { _eAnimationSpeed = value; }
        }

        /// <summary>Percentage of size of sprite height/width to track height/width [min .05 - max .2]</summary>
        [Browsable(false),
        Description("Percentage of size of sprite width to track width [min .05 - max .2]")]
        public float AnimationSize
        {
            get { return _fAnimationSize; }
            set
            {
                if (value < .02f)
                    _fAnimationSize = .02f;
                else if (value < .2f)
                    _fAnimationSize = .2f;
                else
                    _fAnimationSize = value;
            }
        }

        /// <summary>Use an image for the slider background</summary>
        [Browsable(true), Category("Appearence"),
        Description("Use an image for the slider background")]
        public new Bitmap BackgroundImage
        {
            get { return _bmpBackground; }
            set
            {
                try
                {
                    if (value != null && value.GetType() == typeof(Bitmap))
                    {
                        if (this.ErrorHandler != null)
                            this.ErrorHandler.Clear();
                        _bmpBackground = value;
                        base.Width = _bmpBackground.Width;
                        base.Height = _bmpBackground.Height;
                        PropertyChange();
                    }
                    else if (value != null && this.DesignMode)
                    {
                        throw new Exception("Invalid BackGroundImage Property Setting: Invalid image type. Base of image must be a Bitmap");
                    }
                    else if (value == null)
                    {
                        if (this.ErrorHandler != null)
                            this.ErrorHandler.Clear();
                        if (_bmpBackground != null)
                            _bmpBackground.Dispose();
                        _bmpBackground = null;
                    }
                }
                catch (Exception ex) { this.ErrorHandler.SetError(this, ex.Message); }
            }
        }

        /// <summary>Modify button accent color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Modify button accent color")]
        public Color ButtonAccentColor
        {
            get { return _clrButtonAccentColor; }
            set
            {
                _clrButtonAccentColor = value;
                PropertyChange();
            }
        }

        /// <summary>Modify button border color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Modify button border color")]
        public Color ButtonBorderColor
        {
            get { return _clrButtonBorderColor; }
            set
            {
                _clrButtonBorderColor = value;
                PropertyChange();
            }
        }

        /// <summary>Modify button base color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Modify button base color")]
        public Color ButtonColor
        {
            get { return _clrButtonColor; }
            set
            {
                _clrButtonColor = value;
                PropertyChange();
            }
        }

        /// <summary>Adjusts the slider buttons corner radius</summary>
        [Browsable(false),
        Description("Adjusts the slider buttons corner radius")]
        public uint ButtonCornerRadius
        {
            get { return (uint)_iButtonCornerRadius; }
            set { _iButtonCornerRadius = (int)value; }
        }

        /// <summary>Modify slider button size</summary>
        [Browsable(true), Category("Appearence"), RefreshProperties(RefreshProperties.All),
        Description("Modify slider button size")]
        public Size ButtonSize
        {
            get { return _szButtonSize; }
            set
            {
                _szButtonSize = value;
                if (this.DesignMode && !this.AutoSize)
                    PropertyChange();
            }
        }

        /// <summary>Set the button style</summary>
        [Browsable(true), Category("Appearence"), RefreshProperties(RefreshProperties.All),
        Description("Set the button style")]
        public ButtonType ButtonStyle
        {
            get { return _eButtonStyle; }
            set
            {

                if (this.FinishedPropRun && this.DesignMode && _eButtonStyle != value)
                {
                    _eButtonStyle = value;
                    DefaultButtonSize(_eButtonStyle);
                }
                else
                {
                    _eButtonStyle = value;
                }
                PropertyChange();
            }
        }

        /// <summary>Returns the property initiated state</summary>
        [Browsable(false),
        Description("Returns the property initiated state")]
        public bool IsInited
        {
            get { return this.Visible && _bPropInit; }
            private set { _bPropInit = this.Visible && value; }
        }

        /// <summary>The number of clicks the slider moves in response to mouse clicks or pageup/pagedown</summary>
        [Browsable(true), Category("Behavior"),
        Description("The number of clicks the slider moves in response to mouse clicks or pageup/pagedown")]
        public int LargeChange
        {
            get { return _iLargeChange; }
            set
            {
                try
                {
                    if (value < 1 && this.DesignMode && this.FinishedPropRun)
                    {
                        throw new Exception("Invalid LargeChange Property Setting: Large change can not be less then 1");
                    }
                    else
                    {
                        if (this.ErrorHandler != null)
                            this.ErrorHandler.Clear();
                        _iLargeChange = value;
                        PropertyChange();
                    }
                }
                catch (Exception ex) { this.ErrorHandler.SetError(this, ex.Message); }
            }
        }

        /// <summary>The maximum value for the position of the slider</summary>
        [Browsable(true), Category("Behavior"),
        Description("The maximum value for the position of the slider")]
        public int Maximum
        {
            get { return _iMaximum; }
            set
            {
                try
                {
                    if (value <= this.Minimum && this.FinishedPropRun)
                    {
                        if (this.DesignMode)
                            throw new Exception("Invalid Maximum Property Setting: Maximum can not be less then the Minimum value setting");
                    }
                    else
                    {
                        if (this.ErrorHandler != null)
                            this.ErrorHandler.Clear();
                        _iMaximum = value;
                        PropertyChange();
                    }
                }
                catch (Exception ex) { this.ErrorHandler.SetError(this, ex.Message); }
                finally
                {
                    if (!this.DesignMode)
                        SliderFlyOut = _eFlyOutOnFocus;
                }
            }
        }

        /// <summary>The maximum Size value for the control</summary>
        [Browsable(true), Category("Behavior"),
        Description("The maximum Size value for the control [private set]")]
        public Size MaxSize
        {
            get { return _szMaxSize; }
            private set { _szMaxSize = value; }
        }

        /// <summary>The minimum value for the position of the slider</summary>
        [Browsable(true), Category("Behavior"),
        Description("The minimum value for the position of the slider")]
        public int Minimum
        {
            get { return _iMinimum; }
            set
            {
                try
                {
                    if (value >= this.Maximum && this.FinishedPropRun)
                    {
                        if (this.DesignMode)
                            throw new Exception("Invalid Minimum Property Setting: Minimum can not be more then the Maximum value setting");
                    }
                    else
                    {
                        if (this.ErrorHandler != null)
                            this.ErrorHandler.Clear();
                        _iMinimum = value;
                        PropertyChange();
                    }
                }
                catch (Exception ex) { this.ErrorHandler.SetError(this, ex.Message); }
            }
        }

        /// <summary>The minimum Size value for the control</summary>
        [Browsable(true), Category("Behavior"),
        Description("The minimum Size value for the control [private set]")]
        public Size MinSize
        {
            get { return _szMinSize; }
            private set { _szMinSize = value; }
        }

        /// <summary>The orientation of the control</summary>
        [Browsable(true), Category("Appearence"), RefreshProperties(RefreshProperties.All),
        Description("The orientation of the control")]
        public Orientation Orientation
        {
            get { return _eOrientation; }
            set
            {
                _eOrientation = value;
                if (this.FinishedPropRun && this.DesignMode && _eOrientation != value)
                {
                    _eOrientation = value;
                    DefaultButtonSize(_eButtonStyle);
                }
                else
                {
                    _eOrientation = value;
                }
                PropertyChange();
            }
        }

        /// <summary>Returns the slider position to a floating point, requires SmoothScrolling set to true</summary>
        [Browsable(false),
        Description("Returns the slider position to a floating point, requires SmoothScrolling set to true")]
        public double PrecisionValue
        {
            get { return IncrementalValue(); }
        }

        /// <summary>Show the slider button only when control is focused or mouse is hovering</summary>
        [Browsable(true), Category("Appearence"),
        Description("Show the slider button only when control is focused or mouse is hovering")]
        public bool ShowButtonOnHover
        {
            get { return _bShowButtonOnHover; }
            set { _bShowButtonOnHover = value; }
        }

        /// <summary>Enable the flyout caption window</summary>
        [Browsable(true), Category("Appearence"),
        Description("Enable the flyout caption window")]
        public FlyOutStyle SliderFlyOut
        {
            get { return _eFlyOutOnFocus; }
            set
            {
                _eFlyOutOnFocus = value;
                if (_eFlyOutOnFocus != FlyOutStyle.None)
                {
                    if (Orientation == Orientation.Horizontal)
                    {
                        this.FlyOutMaxDepth = 14;
                        this.FlyOutSpacer = 6;
                        if (this.Maximum < 10)
                        {
                            this.FlyOutMaxWidth = 10;
                            this.TrackPadding = 4;
                        }
                        else if (this.Maximum < 100)
                        {
                            this.FlyOutMaxWidth = 20;
                            this.TrackPadding = 6;
                        }
                        else if (this.Maximum < 1000)
                        {
                            this.FlyOutMaxWidth = 30;
                            this.TrackPadding = 12;
                        }
                        else if (this.Maximum > 999)
                        {
                            // probably time
                            this.FlyOutMaxWidth = 54;
                            this.TrackPadding = 24;
                        }
                    }
                    else
                    {
                        if (this.Minimum < 0)
                        {
                            // max 2 digit vertical
                            this.FlyOutSpacer = 2;
                            this.FlyOutMaxDepth = 30;
                            this.FlyOutMaxWidth = 20;
                            this.TrackPadding = 12;
                        }
                        else
                        {
                            // max 2 digit vertical
                            this.FlyOutSpacer = 2;
                            this.FlyOutMaxDepth = 20;
                            this.FlyOutMaxWidth = 18;
                            this.TrackPadding = 5;
                        }
                    }
                }
                else
                {
                    this.FlyOutMaxDepth = 0;
                    this.TrackPadding = 2;
                }
                PropertyChange();
            }
        }

        /// <summary>The number of positions the slider movers in response to arrow keys</summary>
        [Browsable(true), Category("Behavior"),
        Description("The number of positions the slider movers in response to arrow keys")]
        public int SmallChange
        {
            get { return _iSmallChange; }
            set
            {
                try
                {
                    if (value < 1 && this.DesignMode && this.FinishedPropRun)
                    {
                        throw new Exception("Invalid SmallChange Property Setting: Small change can not be less then 1");
                    }
                    else
                    {
                        if (this.ErrorHandler != null)
                            this.ErrorHandler.Clear();
                        _iSmallChange = value;
                        PropertyChange();
                    }
                }
                catch (Exception ex) { this.ErrorHandler.SetError(this, ex.Message); }
            }
        }

        /// <summary>Run the animation effect when focused</summary>
        [Browsable(true), Category("Behavior"),
        Description("Enable smooth scrolling style")]
        public bool SmoothScrolling
        {
            get { return _bSmoothScrolling; }
            set { _bSmoothScrolling = value; }
        }

        /// <summary>Modify slider tick color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Modify slider tick color")]
        public Color TickColor
        {
            get { return _clrTickColor; }
            set
            {
                _clrTickColor = value;
                PropertyChange();
            }
        }

        /// <summary>Select the tickstyle</summary>
        [Browsable(true), Category("Appearence"),
        Description("Select the tickstyle")]
        public TickStyle TickStyle
        {
            get { return _eTickStyle; }
            set
            {
                _eTickStyle = value;
                PropertyChange();
            }
        }

        /// <summary>Select the tick drawing style</summary>
        [Browsable(true), Category("Appearence"),
        Description("Select the tick drawing style")]
        public TickMode TickType
        {
            get { return _eTickType; }
            set
            {
                _eTickType = value;
                PropertyChange();
            }
        }

        /// <summary>Modify slider border color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Modify slider border color")]
        public Color TrackBorderColor
        {
            get { return _clrTrackBorderColor; }
            set
            {
                _clrTrackBorderColor = value;
                PropertyChange();
            }
        }

        /// <summary>Adjust the slider track depth</summary>
        [Browsable(true), Category("Appearence"), RefreshProperties(RefreshProperties.All),
        Description("Adjust the slider track depth")]
        public int TrackDepth
        {
            get { return _iTrackDepth; }
            set
            {
                _iTrackDepth = value;
                if (this.DesignMode && !this.AutoSize)
                    PropertyChange();
            }
        }

        /// <summary>Set the track fill color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Set the track fill color")]
        public Color TrackFillColor
        {
            get { return _clrTrackFillColor; }
            set
            {
                _clrTrackFillColor = value;
                PropertyChange();
            }
        }

        /// <summary>Set the track progress color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Set the track progress color")]
        public Color TrackProgressColor
        {
            get { return _clrTrackProgressColor; }
            set
            {
                _clrTrackProgressColor = value;
                PropertyChange();
            }
        }

        /// <summary>Enable track border shadow</summary>
        [Browsable(true), Category("Appearence"),
        Description("Enable track shadow")]
        public bool TrackShadow
        {
            get { return _bTrackShadow; }
            set
            {
                _bTrackShadow = value;
                PropertyChange();
            }
        }

        /// <summary>Modify track shadow color</summary>
        [Browsable(true), Category("Appearence"),
        Description("Modify track shadow color")]
        public Color TrackShadowColor
        {
            get { return _clrTrackShadowColor; }
            set
            {
                _clrTrackShadowColor = value;
                PropertyChange();
            }
        }

        /// <summary>Modify the display style of track</summary>
        [Browsable(true), Category("Appearence"),
        Description("Modify the display style of track")]
        public TrackType TrackStyle
        {
            get { return _eTrackStyle; }
            set
            {
                _eTrackStyle = value;
                PropertyChange();
            }
        }

        /// <summary>The position of the slider</summary>
        [Browsable(true), Category("Behavior"),
        Description("The position of the slider")]
        public int Value
        {
            get { return _iValue; }
            set
            {
                this.IsInited = !this.DesignMode;
                try
                {
                    if (value > this.Maximum)
                    {
                        if (this.DesignMode && this.FinishedPropRun)
                            throw new Exception("Invalid Value Property Setting: Value can not be more then Maximum setting");
                        else
                            _iValue = this.Maximum;
                    }
                    else if (value < this.Minimum)
                    {
                        if (this.DesignMode && this.FinishedPropRun)
                            throw new Exception("Invalid Value Property Setting: Value can not be less then Minimum setting");
                        else
                            _iValue = this.Minimum;
                    }
                    else
                    {
                        if (this.ErrorHandler != null)
                            this.ErrorHandler.Clear();
                        _iValue = value;
                    }

                    if (this.DesignMode)
                        _dButtonPosition = IncrementalValue();

                    if (!this.FinishedPropRun)
                    {
                        PropertyChange();
                        this.FinishedPropRun = true;
                    }
                    else if (!this.DesignMode)
                    {
                        _dButtonPosition = IncrementalValue();
                        if (ValueChanged != null)
                            ValueChanged(this, new EventArgs());
                        DrawSlider();
                    }
                }
                catch (Exception ex) { this.ErrorHandler.SetError(this, ex.Message); }
            }
        }
        #endregion
        #endregion

        #region Overrides
        protected override void OnCreateControl()
        {
            base.OnCreateControl();
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            Init();
            base.OnHandleCreated(e);
        }

        protected override void OnHandleDestroyed(EventArgs e)
        {
            DeInit();
            base.OnHandleDestroyed(e);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            if (_bHasInitialized)
                DrawSlider();
            base.OnPaint(e);
        }

        protected override void OnBackColorChanged(EventArgs e)
        {
            Init();
            base.OnBackColorChanged(e);
        }

        protected override void OnGotFocus(EventArgs e)
        {
            if (!this.InTarget)
            {
                _eSliderState = SliderSelectedState.Focused;
                DrawSlider();
            }
            base.OnGotFocus(e);
        }

        protected override void OnLostFocus(EventArgs e)
        {
            _eSliderState = SliderSelectedState.None;
            DrawSlider();
            base.OnLostFocus(e);
        }

        protected override void OnMouseClick(MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
            {
				HitTest tst = SliderHitTest();
                if (tst == HitTest.Track || tst == HitTest.Nowhere)
                {
                    int pos;
                    if (Orientation == Orientation.Horizontal)
                        pos = e.X;
                    else
                        pos = e.Y;
                    if (this.SmoothScrolling)
                    {
                        ScrollThis(pos);
                    }
                    else
                    {
                        if (pos < IncrementalValue())
                            ScrollChange(ChangeType.Large, (Orientation == Orientation.Horizontal));
                        else
                            ScrollChange(ChangeType.Large, (Orientation != Orientation.Horizontal));
                    }
                }
            }
            base.OnMouseClick(e);
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            this.InTarget = (SliderHitTest() == HitTest.Button);
            if (this.InTarget)
                _eSliderState = SliderSelectedState.Pressed;
            else
                _eSliderState = SliderSelectedState.Focused;
            DrawSlider();
            base.OnMouseDown(e);
        }

        protected override void OnMouseHover(EventArgs e)
        {
            _eSliderState = SliderSelectedState.Hover;
            DrawSlider();
            base.OnMouseHover(e);
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            _eSliderState = SliderSelectedState.Depressed;
            _bInTarget = false;
            DrawSlider();
            base.OnMouseUp(e);
        }

        protected override void OnMouseLeave(EventArgs e)
        {
            _eSliderState = SliderSelectedState.MouseLeave;
            DrawSlider();
            base.OnMouseLeave(e);
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left && _bInTarget)
            {
                if (Orientation == Orientation.Horizontal)
                    ScrollThis(e.X);
                else
                    ScrollThis(e.Y);
            }
            base.OnMouseMove(e);
        }

        protected override void OnResize(EventArgs e)
        {
            PropertyChange();
            base.OnResize(e);
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            switch (keyData)
            {
                case Keys.Left:
                    ScrollChange(ChangeType.Small, true);
                    DrawSlider();
                    return true;
                case Keys.Right:
                    ScrollChange(ChangeType.Small, false);
                    DrawSlider();
                    return true;
                case Keys.Up:
                    ScrollChange(ChangeType.Small, false);
                    DrawSlider();
                    return true;
                case Keys.Down:
                    ScrollChange(ChangeType.Small, true);
                    DrawSlider();
                    return true;
                case Keys.Home:
                    this.Value = this.Minimum;
                    return true;
                case Keys.End:
                    this.Value = this.Maximum;
                    return true;
            }
            return base.ProcessCmdKey(ref msg, keyData);
        }
        #endregion

        #region Methods
        #region Graphics
        #region Drawing
        /// <summary>Drawing hub</summary>
        private void DrawSlider()
        {
            Rectangle bounds = new Rectangle(0, 0, this.Width, this.Height);
            using (Graphics g = Graphics.FromHdc(_cControlDc.Hdc))
            {
                DrawTrack();
                if (this.SliderFlyOut == FlyOutStyle.Persistant)
                    DrawFlyOut(g);

                switch (_eSliderState)
                {
                    case SliderSelectedState.None:
                        {
                            if (_bPulseTimerOn)
                                StopPulseTimer();
                            if (!ShowButtonOnHover)
                                DrawButton(g, bounds, 1f);
                            break;
                        }
                    case SliderSelectedState.Disabled:
                        {
                            DrawButton(g, bounds, -1f);
                            break;
                        }
                    case SliderSelectedState.MouseLeave:
                        {
                            if (!ShowButtonOnHover || _bPulseTimerOn)
                            {
                                if (this.Focused)
                                    DrawButton(g, bounds, 1.2f);
                                else
                                    DrawButton(g, bounds, 1f);
                            }
                            break;
                        }
                    case SliderSelectedState.Hover:
                        {
                            if (this.SliderFlyOut == FlyOutStyle.OnFocus)
                                DrawFlyOut(g);
                            DrawButton(g, bounds, 1.2f);
                            break;
                        }
                    case SliderSelectedState.Depressed:
                    case SliderSelectedState.Focused:
                        {
                            if (!_bPulseTimerOn)
                            {
                                if (this.SliderFlyOut == FlyOutStyle.OnFocus)
                                    DrawFlyOut(g);
                                DrawButton(g, bounds, 1.2f);
                                if (this.Animated)
                                    StartPulseTimer();
                            }
                            else if (!this.InTarget)
                            {
                                DrawButton(g, bounds, 1.0f);
                            }
                            break;
                        }
                    case SliderSelectedState.Pressed:
                        {
                            if (_bPulseTimerOn)
                                StopPulseTimer();
                            if (this.SliderFlyOut == FlyOutStyle.OnFocus)
                                DrawFlyOut(g);
                            DrawButton(g, bounds, .9f);
                            break;
                        }
                }
            }

            if (!_bPulseTimerOn)
            {
                // draw buffer to control
                using (Graphics g = Graphics.FromHwnd(this.Handle))
                {
                    BitBlt(g.GetHdc(), 0, 0, _cControlDc.Width, _cControlDc.Height, _cControlDc.Hdc, 0, 0, 0xCC0020);
                    g.ReleaseHdc();
                }
                RECT r = new RECT(0, 0, this.Width, this.Height);
                ValidateRect(this.Handle, ref r);
            }
        }

        /// <summary>Backfill the buffer</summary>
        private void DrawBackGround(Graphics g, Rectangle bounds)
        {
            using (Brush br = new SolidBrush(this.BackColor))
                g.FillRectangle(br, bounds);
        }

        /// <summary>Adjust gamma of an image [not used]</summary>
        private void DrawBrightImage(Graphics g, Image img, Rectangle bounds, float gamma)
        {
            try
            {
                using (Bitmap buttonImage = new Bitmap(img))
                {
                    using (ImageAttributes imageAttr = new ImageAttributes())
                    {
                        if (gamma > .9f)
                            gamma = .9f;
                        if (gamma < .2f)
                            gamma = .2f;
                        // raise gamma
                        imageAttr.SetGamma(gamma);
                        g.DrawImage(buttonImage,
                            bounds,
                            0, 0,
                            buttonImage.Width,
                            buttonImage.Height,
                            GraphicsUnit.Pixel,
                            imageAttr);
                    }
                }
            }
            catch { }
        }

        private void DrawButton(Graphics g, Rectangle bounds, float level)
        {
            Rectangle buttonRect = GetButtonRectangle();
            if (level != 1f)
            {
                using (ImageAttributes ia = new ImageAttributes())
                {
                    ColorMatrix cm = new ColorMatrix();
                    if (level == -1)
                    {
                        cm.Matrix00 = 1f;//r
                        cm.Matrix11 = 1f;//g
                        cm.Matrix22 = 1f;//b
                        cm.Matrix33 = .7f;//a
                        cm.Matrix44 = 1f;//w
                    }
                    else
                    {
                        cm.Matrix00 = level;
                        cm.Matrix11 = level;
                        cm.Matrix22 = level;
                        cm.Matrix33 = 1f;
                        cm.Matrix44 = 1f;
                    }
                    ia.SetColorMatrix(cm);
                    g.DrawImage(_bmpButton, buttonRect, 0, 0, _bmpButton.Width, _bmpButton.Height, GraphicsUnit.Pixel, ia);
                }
            }
            else
            {
                DrawImage(g, _bmpButton, buttonRect);
            }
        }

        /// <summary>Draw a disabled image using the control</summary>
        private void DrawDisabledImage(Graphics g, Image image, Rectangle bounds)
        {
            ControlPaint.DrawImageDisabled(g, image, bounds.X, bounds.Y, Color.Transparent);
        }

        /// <summary>Draw an unaltered image</summary>
        private void DrawImage(Graphics g, Image image, Rectangle bounds)
        {
            g.DrawImage(image, bounds);
        }

        /// <summary>Draw the slider ticks</summary>
        private void DrawTicks(Graphics g, Rectangle bounds)
        {
            Rectangle trackRect = GetTrackRectangle();
            float increment = (float)Increment();
            increment = (float)Increment();
            int count = (int)(IncrementScale());

            float endcap = (Orientation == Orientation.Horizontal ? (float)trackRect.Right - (1 + this.ButtonSize.Width / 2) : (float)trackRect.Bottom - (1 + this.ButtonSize.Height / 2));
            float offset = 0;
            int shadowlen = this.TickMaxLength - 1;
            int spacer = this.TickMaxLength + this.TickMinPadding;
            RectangleF buttonRect = GetButtonRectangle();

            switch (this.TickType)
            {
                #region Composite Style
                case TickMode.Composite:
                    {
                        using (GraphicsMode md = new GraphicsMode(g, SmoothingMode.None))
                        {
                            switch (this.TickStyle)
                            {
                                case TickStyle.Both:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;

                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(val, top), new PointF(val, top + this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, top + 1), new PointF(val + 1, top + shadowlen));
                                                        g.DrawLine(pn2, new PointF(val, bottom), new PointF(val, bottom - this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, bottom), new PointF(val + 1, bottom - shadowlen));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(val, top), new PointF(val, top + 2));
                                                        g.DrawLine(pn, new PointF(val, bottom), new PointF(val, bottom - 2));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(left, val), new PointF(left + this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(left + 1, val + 1), new PointF(left + shadowlen, val + 1));
                                                        g.DrawLine(pn2, new PointF(right, val), new PointF(right - this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(right, val + 1), new PointF(right - shadowlen, val + 1));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(left, val), new PointF(left + 2, val));
                                                        g.DrawLine(pn, new PointF(right, val), new PointF(right - 2, val));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.BottomRight:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(val, bottom), new PointF(val, bottom - this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, bottom), new PointF(val + 1, bottom - shadowlen));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(val, bottom), new PointF(val, bottom - 2));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(right, val), new PointF(right - this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(right, val + 1), new PointF(right - shadowlen, val + 1));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(right, val), new PointF(right - 2, val));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.TopLeft:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(val, top), new PointF(val, top + this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, top + 1), new PointF(val + 1, top + shadowlen));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(val, top), new PointF(val, top + 2));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(left, val), new PointF(left + this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(left + 1, val + 1), new PointF(left + shadowlen, val + 1));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(left, val), new PointF(left + 2, val));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                            }
                        }
                        break;
                    }
                #endregion

                #region Large Stepped Style
                case TickMode.LargeStepped:
                    {
                        using (GraphicsMode md = new GraphicsMode(g, SmoothingMode.None))
                        {
                            switch (this.TickStyle)
                            {
                                case TickStyle.Both:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        val = (increment * i) + offset;
                                                        g.DrawLine(pn2, new PointF(val, top), new PointF(val, top + this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, top + 1), new PointF(val + 1, top + shadowlen));
                                                        g.DrawLine(pn2, new PointF(val, bottom), new PointF(val, bottom - this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, bottom), new PointF(val + 1, bottom - shadowlen));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        val = (increment * i) + offset;
                                                        g.DrawLine(pn2, new PointF(left, val), new PointF(left + this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(left + 1, val + 1), new PointF(left + shadowlen, val + 1));
                                                        g.DrawLine(pn2, new PointF(right, val), new PointF(right - this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(right, val + 1), new PointF(right - shadowlen, val + 1));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.BottomRight:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        val = (increment * i) + offset;
                                                        g.DrawLine(pn2, new PointF(val, bottom), new PointF(val, bottom - this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, bottom), new PointF(val + 1, bottom - shadowlen));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(right, val), new PointF(right - this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(right, val + 1), new PointF(right - shadowlen, val + 1));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.TopLeft:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(val, top), new PointF(val, top + this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, top + 1), new PointF(val + 1, top + shadowlen));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(left, val), new PointF(left + this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(left + 1, val + 1), new PointF(left + shadowlen, val + 1));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                            }
                        }
                        break;
                    }
                #endregion

                #region Precision Style
                case TickMode.Precision:
                    {
                        using (GraphicsMode md = new GraphicsMode(g, SmoothingMode.None))
                        {
                            float split = increment * .5f;
                            bool valid = split > 2;
                            switch (this.TickStyle)
                            {
                                case TickStyle.Both:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;

                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(val, top), new PointF(val, top + this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, top + 1), new PointF(val + 1, top + shadowlen));
                                                        g.DrawLine(pn2, new PointF(val, bottom), new PointF(val, bottom - this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, bottom), new PointF(val + 1, bottom - shadowlen));
                                                        if (valid && val < endcap)
                                                        {
                                                            g.DrawLine(pn, new PointF(val + split, top), new PointF(val + split, top + 1));
                                                            g.DrawLine(pn, new PointF(val + split, bottom), new PointF(val + split, bottom - 1));
                                                        }
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(val, top), new PointF(val, top + 2));
                                                        g.DrawLine(pn, new PointF(val, bottom), new PointF(val, bottom - 2));
                                                        if (valid && val < endcap)
                                                        {
                                                            g.DrawLine(pn, new PointF(val + split, top), new PointF(val + split, top + 1));
                                                            g.DrawLine(pn, new PointF(val + split, bottom), new PointF(val + split, bottom - 1));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(left, val), new PointF(left + this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(left + 1, val + 1), new PointF(left + shadowlen, val + 1));
                                                        g.DrawLine(pn2, new PointF(right, val), new PointF(right - this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(right, val + 1), new PointF(right - shadowlen, val + 1));
                                                        if (valid && val < endcap)
                                                        {
                                                            g.DrawLine(pn, new PointF(left, val + split), new PointF(left + 1, val + split));
                                                            g.DrawLine(pn, new PointF(right, val + split), new PointF(right - 1, val + split));
                                                        }
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(left, val), new PointF(left + 2, val));
                                                        g.DrawLine(pn, new PointF(right, val), new PointF(right - 2, val));
                                                        if (valid && val < endcap)
                                                        {
                                                            g.DrawLine(pn, new PointF(left, val + split), new PointF(left + 1, val + split));
                                                            g.DrawLine(pn, new PointF(right, val + split), new PointF(right - 1, val + split));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.BottomRight:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(val, bottom), new PointF(val, bottom - this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, bottom), new PointF(val + 1, bottom - shadowlen));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(val + split, bottom), new PointF(val + split, bottom - 1));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(val, bottom), new PointF(val, bottom - 2));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(val + split, bottom), new PointF(val + split, bottom - 1));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(right, val), new PointF(right - this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(right, val + 1), new PointF(right - shadowlen, val + 1));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(right, val + split), new PointF(right - 2, val + split));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(right, val), new PointF(right - 2, val));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(right, val + split), new PointF(right - 2, val + split));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.TopLeft:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(val, top), new PointF(val, top + this.TickMaxLength));
                                                        g.DrawLine(pn3, new PointF(val + 1, top + 1), new PointF(val + 1, top + shadowlen));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(val + split, top), new PointF(val + split, top + 1));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(val, top), new PointF(val, top + 2));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(val + split, top), new PointF(val + split, top + 1));
                                                    }
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    if (Mod(i, this.LargeChange))
                                                    {
                                                        g.DrawLine(pn2, new PointF(left, val), new PointF(left + this.TickMaxLength, val));
                                                        g.DrawLine(pn3, new PointF(left + 1, val + 1), new PointF(left + shadowlen, val + 1));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(left, val + split), new PointF(left + 2, val + split));
                                                    }
                                                    else
                                                    {
                                                        g.DrawLine(pn, new PointF(left, val), new PointF(left + 2, val));
                                                        if (valid && val < endcap)
                                                            g.DrawLine(pn, new PointF(left, val + split), new PointF(left + 2, val + split));
                                                    }
                                                }
                                            }
                                        }
                                        break;
                                    }
                            }
                        }
                        break;
                    }
                #endregion

                #region Standard Tick Style
                case TickMode.Standard:
                    {
                        using (GraphicsMode md = new GraphicsMode(g, SmoothingMode.None))
                        {
                            switch (this.TickStyle)
                            {
                                case TickStyle.Both:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    g.DrawLine(pn, new PointF(val, top), new PointF(val, top + 2));
                                                    g.DrawLine(pn, new PointF(val, bottom), new PointF(val, bottom - 2));
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    g.DrawLine(pn, new PointF(left, val), new PointF(left + 2, val));
                                                    g.DrawLine(pn, new PointF(right, val), new PointF(right - 2, val));
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.BottomRight:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float bottom = buttonRect.Bottom + spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    g.DrawLine(pn, new PointF(val, bottom), new PointF(val, bottom - 2));
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float right = buttonRect.Right + spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    g.DrawLine(pn, new PointF(right, val), new PointF(right - 2, val));
                                                }
                                            }
                                        }
                                        break;
                                    }
                                case TickStyle.TopLeft:
                                    {
                                        if (this.Orientation == Orientation.Horizontal)
                                        {
                                            float top = buttonRect.Top - spacer;
                                            offset = (this.ButtonSize.Width / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    g.DrawLine(pn, new PointF(val, top), new PointF(val, top + 2));
                                                }
                                            }
                                        }
                                        else
                                        {
                                            float left = buttonRect.Left - spacer;
                                            offset = (this.ButtonSize.Height / 2) + this.TrackPadding;
                                            float val = offset;
                                            using (Pen pn = new Pen(this.TickColor, .5f), pn2 = new Pen(this.TickColor, 1f), pn3 = new Pen(Color.FromArgb(100, Color.DarkGray), 1f))
                                            {
                                                for (int i = 0; i < count + 1; i++)
                                                {
                                                    val = (increment * i) + offset;
                                                    g.DrawLine(pn, new PointF(left, val), new PointF(left + 2, val));
                                                }
                                            }
                                        }
                                        break;
                                    }
                            }
                        }
                        break;
                    }
                #endregion
            }
        }

        /// <summary>Draw slider and background dc</summary>
        private void DrawTrack()
        {
            BitBlt(_cControlDc.Hdc, 0, 0, _cTrackDc.Width, _cTrackDc.Height, _cTrackDc.Hdc, 0, 0, 0xCC0020);
            if (TrackStyle == TrackType.Progress)
            {
                Rectangle trackRect = GetTrackRectangle();
                Rectangle buttonRect = GetButtonRectangle();
                int length;

                trackRect.Inflate(-1, -1);
                if (Orientation == Orientation.Horizontal)
                {
                    if (_iValue == _iMinimum)
                    {
                        length = 0;
                    }
                    else if (_iValue == _iMaximum)
                    {
                        if (this.SmoothScrolling)
                        {
                            length = buttonRect.Right - (trackRect.Left + 1);
                            trackRect.Width = length;
                        }
                        else
                        {
                            length = buttonRect.Right - (trackRect.Left + 2);
                            trackRect.Width = length;
                        }
                    }
                    else
                    {
                        length = buttonRect.Right - (trackRect.Left + (int)(buttonRect.Width * .5f));
                        trackRect.Width = length;
                    }
                }
                else
                {
                    if (_iValue == _iMinimum)
                    {
                        length = 0;
                    }
                    else if (_iValue == _iMaximum)
                    {
                        if (this.SmoothScrolling)
                        {
                            length = trackRect.Bottom - (buttonRect.Top + 1);
                            trackRect.Y = buttonRect.Top - 1;
                            trackRect.Height = length;
                        }
                        else
                        {
                            length = trackRect.Bottom - (buttonRect.Top + 3);
                            trackRect.Height = length;
                        }
                    }
                    else
                    {
                        length = trackRect.Bottom - (buttonRect.Top + (int)(buttonRect.Height * .5f));
                        trackRect.Y = buttonRect.Top + (int)(buttonRect.Height * .5f) - 2;
                        trackRect.Height = length;
                    }
                }
                if (length > 1)
                {
                    using (Graphics g = Graphics.FromHdc(_cControlDc.Hdc))
                    {
                        using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                        {
                            using (GraphicsPath gp = CreateRoundRectanglePath(g, trackRect, 2))
                            {
                                using (LinearGradientBrush fillBrush = new LinearGradientBrush(
                                    buttonRect,
                                    Color.FromArgb(120, Color.White),
                                    Color.FromArgb(250, this.TrackProgressColor),
                                    (Orientation == Orientation.Horizontal) ? LinearGradientMode.Vertical : LinearGradientMode.Horizontal))
                                {
                                    Blend blnd = new Blend();
                                    blnd.Positions = new float[] { 0f, .5f, 1f };
                                    blnd.Factors = new float[] { .5f, .7f, .3f };
                                    fillBrush.Blend = blnd;
                                    g.FillPath(fillBrush, gp);
                                }
                            }
                        }
                    }
                }
            }
        }
        #endregion

        #region Graphics Creation
        /// <summary>Create the button bitmap</summary>
        private void CreateButtonBitmap()
        {
            Rectangle buttonRect = GetButtonRectangle();
            float fx;
            float fy;

            buttonRect.X = 0;
            buttonRect.Y = 0;
            Rectangle accentRect = buttonRect;

            _bmpButton = new Bitmap(buttonRect.Width + 1, buttonRect.Height + 1);

            switch (this.ButtonStyle)
            {
                #region Precision
                case ButtonType.PointerUpRight:
                case ButtonType.PointerDownLeft:
                    {
                        using (Graphics g = Graphics.FromImage(_bmpButton))
                        {
                            using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                            {
                                int offset = (int)(buttonRect.Height * .2);
                                buttonRect.Inflate(0, -offset);
                                buttonRect.Y = 0;

                                using (GraphicsPath gp = CreatePointedRectangularPath(g, buttonRect, PointDirection.Bottom, 3, offset * 2, -1))
                                {
                                    using (Brush br = new SolidBrush(Color.LightGray))
                                        g.FillPath(br, gp);
                                    using (LinearGradientBrush fillBrush = new LinearGradientBrush(
                                        buttonRect,
                                        this.ButtonAccentColor,
                                        this.ButtonColor,
                                        LinearGradientMode.Horizontal))
                                    {
                                        Blend blnd = new Blend();
                                        blnd.Positions = new float[] { 0f, .5f, 1f };
                                        blnd.Factors = new float[] { .2f, .8f, .2f };
                                        fillBrush.Blend = blnd;
                                        g.FillPath(fillBrush, gp);
                                    }
                                    using (Pen borderPen = new Pen(Color.FromArgb(180, this.ButtonBorderColor), .5f))
                                        g.DrawPath(borderPen, gp);
                                }
                            }
                        }
                        if (this.ButtonStyle == ButtonType.PointerUpRight)
                        {
                            if (this.ButtonStyle == ButtonType.PointerUpRight)
                                _bmpButton.RotateFlip(RotateFlipType.Rotate180FlipX);
                        }
                        if (Orientation == Orientation.Vertical)
                            _bmpButton.RotateFlip(RotateFlipType.Rotate90FlipNone);
                        break;
                    }
                #endregion

                #region Round
                // round button style
                case ButtonType.Round:
                    {
                        using (Graphics g = Graphics.FromImage(_bmpButton))
                        {
                            using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                            {
                                using (GraphicsPath gp = new GraphicsPath())
                                {
                                    gp.AddEllipse(buttonRect);
                                    // fill with base color
                                    using (Brush br = new SolidBrush(Color.FromArgb(255, this.ButtonColor)))
                                        g.FillPath(br, gp);
                                    // add top sheen
                                    using (LinearGradientBrush fillBrush = new LinearGradientBrush(
                                        buttonRect,
                                        Color.FromArgb(180, Color.White),
                                        this.ButtonColor,
                                        LinearGradientMode.Vertical))
                                    {
                                        Blend blnd = new Blend();
                                        blnd.Positions = new float[] { 0f, .1f, .2f, .3f, .6f, 1f };
                                        blnd.Factors = new float[] { .2f, .3f, .4f, .5f, 1f, 1f };
                                        fillBrush.Blend = blnd;
                                        g.FillPath(fillBrush, gp);
                                    }
                                    // add the bottom glow
                                    using (PathGradientBrush borderBrush = new PathGradientBrush(gp))
                                    {
                                        using (GraphicsPath ga = new GraphicsPath())
                                        {
                                            accentRect.Inflate(0, (int)-(accentRect.Height * .2f));
                                            accentRect.Offset(0, (int)(ButtonSize.Width * .2f));
                                            ga.AddEllipse(accentRect);
                                            // center focus
                                            fx = accentRect.Width * .5f;
                                            fy = accentRect.Height * 1f;
                                            borderBrush.CenterColor = this.ButtonColor;
                                            borderBrush.SurroundColors = new Color[] { this.ButtonAccentColor };
                                            borderBrush.FocusScales = new PointF(1f, 0f);
                                            borderBrush.CenterPoint = new PointF(fx, fy);
                                            g.FillPath(borderBrush, ga);
                                        }
                                        // spotight offsets
                                        fx = buttonRect.Width * .2f;
                                        fy = buttonRect.Height * .05f;
                                        // draw the spotlight
                                        borderBrush.CenterColor = Color.FromArgb(120, Color.White);
                                        borderBrush.SurroundColors = new Color[] { Color.FromArgb(5, Color.Silver) };
                                        borderBrush.FocusScales = new PointF(.2f, .2f);
                                        borderBrush.CenterPoint = new PointF(fx, fy);
                                        g.FillPath(borderBrush, gp);
                                    }
                                    // draw the border
                                    using (Pen borderPen = new Pen(this.ButtonBorderColor, .5f))
                                        g.DrawPath(borderPen, gp);
                                }
                            }
                        }
                        break;
                    }
                #endregion

                #region Hybrid
                case ButtonType.Hybrid:
                    {
                        using (Graphics g = Graphics.FromImage(_bmpButton))
                        {
                            using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                            {
                                using (GraphicsPath gp = new GraphicsPath())
                                {
                                    gp.AddEllipse(buttonRect);
                                    using (PathGradientBrush borderBrush = new PathGradientBrush(gp))
                                    {
                                        // center focus
                                        fx = buttonRect.Width * .5f;
                                        fy = buttonRect.Height * .5f;
                                        borderBrush.CenterColor = this.ButtonColor;
                                        borderBrush.SurroundColors = new Color[] { this.ButtonAccentColor };
                                        borderBrush.FocusScales = new PointF(.5f, .5f);
                                        borderBrush.CenterPoint = new PointF(fx, fy);
                                        g.FillPath(borderBrush, gp);

                                    }// draw the border
                                    using (Pen borderPen = new Pen(this.ButtonBorderColor, .5f))
                                        g.DrawPath(borderPen, gp);
                                }
                            }
                        }
                        break;
                    }
                #endregion

                #region Rounded Rectangle
                case ButtonType.RoundedRectInline:
                case ButtonType.RoundedRectOverlap:
                    {
                        using (Graphics g = Graphics.FromImage(_bmpButton))
                        {
                            using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                            {
                                using (GraphicsPath gp = CreateRoundRectanglePath(g, buttonRect, this.ButtonCornerRadius))
                                {
                                    // fill with solid base color
                                    using (Brush br = new SolidBrush(Color.FromArgb(255, this.ButtonColor)))
                                        g.FillPath(br, gp);
                                    fx = buttonRect.Width * .5f;
                                    fy = buttonRect.Height * .5f;
                                    // add a shine
                                    LinearGradientMode md;
                                    if (Orientation == Orientation.Horizontal)
                                    {
                                        if (this.ButtonStyle == ButtonType.RoundedRectOverlap)
                                            md = LinearGradientMode.Horizontal;
                                        else
                                            md = LinearGradientMode.Vertical;
                                    }
                                    else
                                    {
                                        if (this.ButtonStyle == ButtonType.RoundedRectOverlap)
                                            md = LinearGradientMode.Vertical;
                                        else
                                            md = LinearGradientMode.Horizontal;
                                    }
                                    // draw it
                                    using (LinearGradientBrush fillBrush = new LinearGradientBrush(
                                        buttonRect,
                                        Color.FromArgb(120, Color.White),
                                        Color.FromArgb(5, Color.Silver),
                                        md))
                                    {
                                        Blend blnd = new Blend();
                                        blnd.Positions = new float[] { 0f, .2f, .4f, .7f, .8f, 1f };
                                        blnd.Factors = new float[] { .2f, .4f, .5f, .4f, .2f, .1f };
                                        fillBrush.Blend = blnd;
                                        g.FillPath(fillBrush, gp);
                                    }
                                    // draw the border
                                    using (Pen borderPen = new Pen(Color.FromArgb(220, this.ButtonBorderColor), .5f))
                                        g.DrawPath(borderPen, gp);
                                    // add a spotlight underneath
                                    accentRect.Offset(0, (int)(accentRect.Height * .6f));
                                    // center focus
                                    if (Orientation == Orientation.Horizontal && this.ButtonStyle == ButtonType.RoundedRectOverlap
                                        || Orientation == Orientation.Vertical && this.ButtonStyle == ButtonType.RoundedRectInline)
                                    {

                                        fx = accentRect.Width * .1f;
                                        fy = accentRect.Height * .5f;
                                        // notch it down a little
                                        accentRect.Offset(0, (int)(accentRect.Height * .2f));
                                    }
                                    else
                                    {
                                        fx = accentRect.Width * .5f;
                                        fy = accentRect.Height * .1f;
                                    }

                                    using (GraphicsPath ga = new GraphicsPath())
                                    {
                                        ga.AddEllipse(accentRect);
                                        // draw bottom glow
                                        using (PathGradientBrush borderBrush = new PathGradientBrush(ga))
                                        {
                                            borderBrush.CenterColor = this.ButtonAccentColor;
                                            borderBrush.SurroundColors = new Color[] { Color.Transparent };
                                            borderBrush.FocusScales = new PointF(.1f, .2f);
                                            borderBrush.CenterPoint = new PointF(fx, fy);
                                            g.FillPath(borderBrush, ga);
                                        }
                                    }
                                    using (GraphicsPath ga = new GraphicsPath())
                                    {
                                        if (this.ButtonStyle == ButtonType.RoundedRectOverlap)
                                            ga.AddEllipse(0, 0, buttonRect.Width, 4);
                                        else
                                            ga.AddEllipse(2, 0, buttonRect.Width - 4, 4);
                                        // spotight offsets
                                        fx = buttonRect.Width * .5f;
                                        fy = buttonRect.Height * .05f;
                                        // draw the top spotlight
                                        using (PathGradientBrush borderBrush = new PathGradientBrush(ga))
                                        {
                                            borderBrush.CenterColor = Color.FromArgb(120, Color.White);
                                            borderBrush.SurroundColors = new Color[] { Color.FromArgb(5, Color.Silver) };
                                            borderBrush.FocusScales = new PointF(.2f, .2f);
                                            borderBrush.CenterPoint = new PointF(fx, fy);
                                            g.FillPath(borderBrush, gp);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                #endregion

                #region Glass
                case ButtonType.GlassInline:
                case ButtonType.GlassOverlap:
                    {
                        using (Graphics g = Graphics.FromImage(_bmpButton))
                        {
                            using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                            {
                                using (GraphicsPath gp = CreateRoundRectanglePath(g, buttonRect, this.ButtonCornerRadius))
                                {
                                    fx = buttonRect.Width * .5f;
                                    fy = buttonRect.Height * .5f;
                                    // add a shine
                                    using (PathGradientBrush borderBrush = new PathGradientBrush(gp))
                                    {
                                        borderBrush.CenterColor = Color.FromArgb(100, Color.DarkGray);
                                        borderBrush.SurroundColors = new Color[] { Color.FromArgb(120, Color.Silver) };
                                        borderBrush.FocusScales = new PointF(1f, .5f);
                                        borderBrush.CenterPoint = new PointF(fx, fy);
                                        g.FillPath(borderBrush, gp);
                                    }
                                    // draw the border
                                    using (Pen borderPen = new Pen(this.ButtonBorderColor, .5f))
                                        g.DrawPath(borderPen, gp);
                                    // add a spotlight underneath
                                    accentRect.Offset(0, (int)(accentRect.Height * .8f));
                                    // center focus
                                    if (Orientation == Orientation.Horizontal && this.ButtonStyle == ButtonType.RoundedRectOverlap
                                        || Orientation == Orientation.Vertical && this.ButtonStyle == ButtonType.RoundedRectInline)
                                    {

                                        fx = accentRect.Width * .05f;
                                        fy = accentRect.Height * .5f;
                                    }
                                    else
                                    {
                                        fx = accentRect.Width * .5f;
                                        fy = accentRect.Height * .05f;
                                    }
                                    using (GraphicsPath ga = new GraphicsPath())
                                    {
                                        ga.AddEllipse(accentRect);
                                        using (PathGradientBrush borderBrush = new PathGradientBrush(ga))
                                        {
                                            borderBrush.CenterColor = Color.FromArgb(120, this.ButtonAccentColor);
                                            borderBrush.SurroundColors = new Color[] { Color.FromArgb(5, Color.Silver) };
                                            borderBrush.FocusScales = new PointF(.2f, .2f);
                                            borderBrush.CenterPoint = new PointF(fx, fy);
                                            g.FillPath(borderBrush, ga);
                                        }
                                    }
                                    using (GraphicsPath ga = new GraphicsPath())
                                    {
                                        ga.AddEllipse(0, 0, buttonRect.Width, 4);
                                        // spotight offsets
                                        fx = buttonRect.Width * .5f;
                                        fy = buttonRect.Height * .05f;
                                        // draw the top spotlight
                                        using (PathGradientBrush borderBrush = new PathGradientBrush(ga))
                                        {
                                            borderBrush.CenterColor = Color.FromArgb(120, Color.White);
                                            borderBrush.SurroundColors = new Color[] { Color.FromArgb(5, Color.Silver) };
                                            borderBrush.FocusScales = new PointF(.2f, .2f);
                                            borderBrush.CenterPoint = new PointF(fx, fy);
                                            g.FillPath(borderBrush, gp);
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                #endregion
            }
            _bmpButton.MakeTransparent();
        }

        /// <summary>Create graphics objects</summary>
        private void CreateGraphicsObjects()
        {
            DestroyGraphicsObjects();
            // load primary buffer
            _cControlDc = new cStoreDc();
            _cControlDc.Height = this.Height;
            _cControlDc.Width = this.Width;
            // track and background dc
            _cTrackDc = new cStoreDc();
            _cTrackDc.Height = this.Height;
            _cTrackDc.Width = this.Width;
            // draw the track
            CreateTrack();
            // create the button image
            CreateButtonBitmap();
            // create the animation sprite
            CreateSprite();
        }

        /// <summary>create the animations sprite</summary>
        private void CreateSprite()
        {
            Rectangle trackRect = GetTrackRectangle();
            int width = (int)(Orientation == Orientation.Horizontal ? trackRect.Width * .1f : trackRect.Height * .1f);
            int height = this.TrackDepth;

            DestroySprite();
            // draw the line sprite into a bmp
            _bmpSprite = new Bitmap(width, height);
            using (Graphics g = Graphics.FromImage(_bmpSprite))
            {
                using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                {
                    g.CompositingMode = CompositingMode.SourceOver;
                    Rectangle imageRect = new Rectangle(0, 0, width, height);
                    // draw sprite
                    using (LinearGradientBrush fillBrush = new LinearGradientBrush(
                        imageRect,
                        Color.White,
                        Color.Transparent,
                        LinearGradientMode.Horizontal))
                    {
                        Blend blnd = new Blend();
                        blnd.Positions = new float[] { 0f, .2f, .5f, .8f, 1f };
                        blnd.Factors = new float[] { 1f, .6f, 0f, .4f, 1f };
                        fillBrush.Blend = blnd;
                        using (GraphicsPath gp = CreateRoundRectanglePath(g, imageRect, 2))
                            g.FillPath(fillBrush, gp);
                    }
                }
            }
            // make transparent
            _bmpSprite.MakeTransparent();
            // rotate
            if (Orientation == Orientation.Vertical)
                _bmpSprite.RotateFlip(RotateFlipType.Rotate90FlipX);
        }

        /// <summary>Create the track dc</summary>
        private void CreateTrack()
        {
            Rectangle bounds = new Rectangle(0, 0, this.Width, this.Height);
            Rectangle trackRect = GetTrackRectangle();

            using (Graphics g = Graphics.FromHdc(_cTrackDc.Hdc))
            {
                // fill it in
                if (this.BackgroundImage != null)
                {
                    using (GraphicsMode md = new GraphicsMode(g, SmoothingMode.HighQuality))
                    {
                        using (ImageAttributes ia = new ImageAttributes())
                            g.DrawImage(this.BackgroundImage, bounds, 0, 0, this.BackgroundImage.Width, this.BackgroundImage.Height, GraphicsUnit.Pixel, ia);
                    }
                }
                else
                {
                    DrawBackGround(g, bounds);
                }
                // draw ticks
                if (this.TickStyle != TickStyle.None)
                    DrawTicks(g, bounds);
                // create the path
                if (Orientation == Orientation.Horizontal)//***offsets wrong on createpath?
                    trackRect.Width -= 1;
                else
                    trackRect.Height -= 1;
                using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                {
                    // draw track shadow
                    if (this.TrackShadow)
                    {
                        trackRect.Inflate(2, 2);
                        using (GraphicsPath gp = CreateRoundRectanglePath(g, trackRect, 2))
                        {
                            // light shadow
                            using (Pen pn = new Pen(Color.FromArgb(80, this.TrackShadowColor), 1f))
                                g.DrawPath(pn, gp);
                        }
                        trackRect.Inflate(-1, -1);
                        using (GraphicsPath gp = CreateRoundRectanglePath(g, trackRect, 2))
                        {
                            // darker
                            using (Pen pn = new Pen(Color.FromArgb(120, this.TrackShadowColor), 1f))
                                g.DrawPath(pn, gp);
                        }
                        trackRect.Inflate(-1, -1);
                    }

                    using (GraphicsPath gp = CreateRoundRectanglePath(g, trackRect, 2))
                    {
                        // fill color
                        if (this.TrackFillColor != Color.Transparent)
                        {
                            using (Brush br = new SolidBrush(this.TrackFillColor))
                                g.FillPath(br, gp);
                        }
                        // draw the outline
                        using (Pen pn = new Pen(this.TrackBorderColor, 1f))
                            g.DrawPath(pn, gp);
                    }
                }
            }
        }

        /// <summary>Destroy grahics objects</summary>
        private void DestroyGraphicsObjects()
        {
            // destroy sprite and timer
            StopPulseTimer();
            DestroySprite();
            // destroy buffers
            if (_cControlDc != null)
                _cControlDc.Dispose();
            if (_cTrackDc != null)
                _cTrackDc.Dispose();
            if (_cAnimationDc != null)
                _cAnimationDc.Dispose();
            // destroy images
            if (_bmpButton != null)
                _bmpButton.Dispose();
            if (_bmpSprite != null)
                _bmpSprite.Dispose();
        }

        /// <summary>Destroy animation sprite</summary>
        private void DestroySprite()
        {
            if (_bmpSprite != null)
                _bmpSprite.Dispose();
        }
        #endregion

        #region Animation
        #region Drawing
        private void DrawPulse()
        {
            if (_cControlDc != null)
            {
                int offset = 0;
                Rectangle buttonRect = GetButtonRectangle();
                Rectangle trackRect = GetTrackRectangle();
                Rectangle lineRect;
                Rectangle clipRect = trackRect;

                // copy unaltered image into buffer
                BitBlt(_cAnimationDc.Hdc, 0, 0, _cAnimationDc.Width, _cAnimationDc.Height, _cControlDc.Hdc, 0, 0, 0xCC0020);

                if (Orientation == Orientation.Horizontal)
                {
                    _iLinePos += 2;
                    if (_iLinePos > trackRect.Right - _bmpSprite.Width)
                        _iLinePos = trackRect.Left + 1;
                    if (_iLinePos < buttonRect.Left)
                    {
                        if (_iLinePos + _bmpSprite.Width > buttonRect.Left)
                            offset = _iLinePos + _bmpSprite.Width - buttonRect.Left;
                        lineRect = new Rectangle(_iLinePos, trackRect.Y, _bmpSprite.Width - offset, _bmpSprite.Height);
                        // draw sprite -horz
                        DrawPulseSprite(_cAnimationDc.Hdc, _bmpSprite, lineRect, .4f);
                    }
                }
                else
                {
                    _iLinePos -= 1;
                    if (_iLinePos < trackRect.Top + _bmpSprite.Height)
                        _iLinePos = trackRect.Bottom - _bmpSprite.Height;
                    if (_iLinePos > buttonRect.Bottom)
                    {
                        if (_iLinePos - _bmpSprite.Height < buttonRect.Bottom)
                            offset = buttonRect.Bottom - (_iLinePos - _bmpSprite.Height);
                        lineRect = new Rectangle(trackRect.X, _iLinePos, _bmpSprite.Width, _bmpSprite.Height - offset);
                        // draw sprite -vert
                        DrawPulseSprite(_cAnimationDc.Hdc, _bmpSprite, lineRect, .4f);
                    }
                }
                // draw to control
                using (Graphics g = Graphics.FromHwnd(this.Handle))
                {
                    BitBlt(g.GetHdc(), 0, 0, _cAnimationDc.Width, _cAnimationDc.Height, _cAnimationDc.Hdc, 0, 0, 0xCC0020);
                    g.ReleaseHdc();
                }
                RECT r = new RECT(0, 0, this.Width, this.Height);
                ValidateRect(this.Handle, ref r);
            }
        }

        /// <summary>Draws the line sprite</summary>
        private void DrawPulseSprite(IntPtr destdc, Bitmap source, Rectangle bounds, float intensity)
        {
            using (Graphics g = Graphics.FromHdc(destdc))
            {
                g.CompositingMode = CompositingMode.SourceOver;
                AlphaBlend(g, source, bounds, intensity);
            }
        }
        #endregion

        #region Timer
        private void StartPulseTimer()
        {
            if (_cPulseTimer == null)
            {
                _bPulseTimerOn = true;
                if (_cAnimationDc != null)
                    _cAnimationDc.Dispose();
                _cAnimationDc = new cStoreDc();
                _cAnimationDc.Width = _cControlDc.Width;
                _cAnimationDc.Height = _cControlDc.Height;
                BitBlt(_cAnimationDc.Hdc, 0, 0, _cControlDc.Width, _cControlDc.Height, _cControlDc.Hdc, 0, 0, 0xCC0020);
                // timer setup
                _cPulseTimer = new FadeTimer(this);
                _cPulseTimer.Tick += new FadeTimer.TickDelegate(_cPulseTimer_Tick);
                _cPulseTimer.Complete += new FadeTimer.CompleteDelegate(_cPulseTimer_Complete);
                _cPulseTimer.Interval = (int)this.AnimationSpeed;
                _cPulseTimer.Fade(FadeTimer.FadeType.Loop);
            }
        }

        private void StopPulseTimer()
        {
            if (_cPulseTimer != null)
            {
                _iLinePos = 0;
                if (_cAnimationDc != null)
                    _cAnimationDc.Dispose();
                // tear down the timer class
                _cPulseTimer.Reset();
                _cPulseTimer.Tick -= _cPulseTimer_Tick;
                _cPulseTimer.Complete -= _cPulseTimer_Complete;
                _cPulseTimer.Dispose();
                _cPulseTimer = null;
                _bPulseTimerOn = false;
            }
        }

        private void _cPulseTimer_Complete(object sender)
        {
            ResetCallback rs = new ResetCallback(StopPulseTimer);
            this.Invoke(rs);
        }

        private void _cPulseTimer_Tick(object sender)
        {
            DrawPulse();
        }

        #endregion
        #endregion

        #region FlyOut Caption
        private void DrawFlyOut(Graphics g)
        {
            Rectangle buttonRect = GetButtonRectangle();
            Rectangle flyoutRect;
            int pos;

            if (Orientation == Orientation.Horizontal)
            {
                pos = buttonRect.Left + (int)(buttonRect.Width * .5f);
                int offset = (int)(this.FlyOutMaxWidth * .5f);
                flyoutRect = new Rectangle((int)pos - offset, buttonRect.Top - (this.FlyOutMaxDepth + this.FlyOutSpacer), this.FlyOutMaxWidth, this.FlyOutMaxDepth);
                offset -= 8;
                using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                {
                    using (GraphicsPath gp = CreatePointedRectangularPath(g, flyoutRect, PointDirection.Bottom, 5, 4, offset))
                    {
                        using (Brush br = new SolidBrush(Color.FromArgb(220, Color.White)))
                            g.FillPath(br, gp);
                        using (Pen pn = new Pen(Color.FromArgb(160, Color.DimGray)))
                            g.DrawPath(pn, gp);

						FlyOutEventArgs flyOutData = new FlyOutEventArgs();
                        if (FlyOutInfo != null)
							FlyOutInfo(this, flyOutData);
						if (flyOutData.text.Length == 0)
							flyOutData.text = this.Value.ToString("0");
                        using (StringFormat sf = new StringFormat())
                        {
                            sf.FormatFlags = StringFormatFlags.NoWrap;
                            sf.Alignment = StringAlignment.Center;
                            sf.LineAlignment = StringAlignment.Center;
                            using (Font ft = new Font("Arial", 8f, FontStyle.Regular))
								g.DrawString(flyOutData.text, ft, Brushes.Black, flyoutRect, sf);
                        }
                    }
                }
            }
            else
            {
                pos = buttonRect.Top + (int)(buttonRect.Height * .5f);
                int offset = (int)(this.FlyOutMaxWidth * .5f);
                flyoutRect = new Rectangle(buttonRect.Left - (this.FlyOutMaxDepth + this.FlyOutSpacer), pos - offset, this.FlyOutMaxDepth, this.FlyOutMaxWidth);
                using (GraphicsMode mode = new GraphicsMode(g, SmoothingMode.HighQuality))
                {
                    using (GraphicsPath gp = CreatePointedRectangularPath(g, flyoutRect, PointDirection.Right, 5, 4, 1))
                    {
                        using (Brush br = new SolidBrush(Color.FromArgb(200, Color.White)))
                            g.FillPath(br, gp);
                        using (Pen pn = new Pen(Color.FromArgb(240, this.ButtonBorderColor)))
                            g.DrawPath(pn, gp);

						FlyOutEventArgs flyOutData = new FlyOutEventArgs();
                        if (FlyOutInfo != null)
							FlyOutInfo(this, flyOutData);
						if (flyOutData.text.Length == 0)
							flyOutData.text = this.Value.ToString("0");
                        flyoutRect.Width -= 4;
                        using (StringFormat sf = new StringFormat())
                        {
                            sf.FormatFlags = StringFormatFlags.NoWrap;
                            sf.Alignment = StringAlignment.Center;
                            sf.LineAlignment = StringAlignment.Center;
                            using (Font ft = new Font("Arial", 8f, FontStyle.Regular))
								g.DrawString(flyOutData.text, ft, Brushes.Black, flyoutRect, sf);
                        }
                    }
                }
            }

        }
        #endregion
        #endregion

        #region Helpers
        /// <summary>AlphaBlend an image, alpha .1-1</summary>
        private void AlphaBlend(Graphics g, Bitmap bmp, Rectangle bounds, float alpha)
        {
            if (alpha > 1f)
                alpha = 1f;
            else if (alpha < .01f)
                alpha = .01f;
            using (ImageAttributes ia = new ImageAttributes())
            {
                ColorMatrix cm = new ColorMatrix();
                cm.Matrix00 = 1f;
                cm.Matrix11 = 1f;
                cm.Matrix22 = 1f;
                cm.Matrix44 = 1f;
                cm.Matrix33 = alpha;
                ia.SetColorMatrix(cm);
                g.DrawImage(bmp, bounds, 0, 0, bmp.Width, bmp.Height, GraphicsUnit.Pixel, ia);
            }
        }

        /// <summary>Maximun size based on control options</summary>
        private void CalculateMaximumSize()
        {
            Size sz = new Size(0, 0);

            if (this.TickStyle != TickStyle.None)
                sz.Height += (this.TickMaxLength + this.TickMinPadding) * (this.TickStyle == TickStyle.Both ? 2 : 1);
            if (this.SliderFlyOut != FlyOutStyle.None)
                sz.Height += this.FlyOutMaxDepth + this.FlyOutSpacer;

            switch (this.ButtonStyle)
            {
                case ButtonType.GlassInline:
                case ButtonType.RoundedRectInline:
                    {
                        sz.Height += TrackMaxDepth + 6;
                        break;
                    }
                case ButtonType.GlassOverlap:
                case ButtonType.RoundedRectOverlap:
                    {
                        sz.Height += TrackMaxDepth * 4;
                        break;
                    }
                case ButtonType.Round:
                case ButtonType.Hybrid:
                    {
                        sz.Height += TrackMaxDepth + 8;
                        break;
                    }
                case ButtonType.PointerDownLeft:
                case ButtonType.PointerUpRight:
                    {
                        sz.Height += TrackMaxDepth * 4;
                        break;
                    }
            }
            if (Orientation == Orientation.Horizontal)
            {
                this.MaxSize = sz;
            }
            else
            {
                Size flip = new Size(sz.Height, sz.Width);
                this.MaxSize = flip;
            }
        }

        /// <summary>Minimum size based on control options</summary>
        private void CalculateMinimumSize()
        {
            Size sz = new Size(0, 0);

            if (this.TickStyle != TickStyle.None)
                sz.Height += (this.TickMaxLength + this.TickMinPadding) * (this.TickStyle == TickStyle.Both ? 2 : 1);
            if (this.SliderFlyOut != FlyOutStyle.None)
                sz.Height += this.FlyOutMaxDepth + this.FlyOutSpacer;

            switch (this.ButtonStyle)
            {
                case ButtonType.GlassInline:
                case ButtonType.RoundedRectInline:
                    {
                        sz.Height += TrackMinDepth + 6;
                        break;
                    }
                case ButtonType.GlassOverlap:
                case ButtonType.RoundedRectOverlap:
                    {
                        sz.Height += TrackMinDepth * 4;
                        break;
                    }
                case ButtonType.Round:
                case ButtonType.Hybrid:
                    {
                        sz.Height += TrackMinDepth + 8;
                        break;
                    }
                case ButtonType.PointerDownLeft:
                case ButtonType.PointerUpRight:
                    {
                        sz.Height += TrackMinDepth * 4;
                        break;
                    }
            }
            if (Orientation == Orientation.Horizontal)
            {
                this.MinSize = sz;
            }
            else
            {
                Size flip = new Size(sz.Height, sz.Width);
                this.MinSize = flip;
            }
        }

        private GraphicsPath CreatePointedRectangularPath(Graphics g, Rectangle bounds, PointDirection direction, float radius, int depth, int inset)
        {
            int diff = 0;
            // create a path
            GraphicsPath pathBounds = new GraphicsPath();
            switch (direction)
            {
                case PointDirection.Bottom:
                    {
                        // line left
                        pathBounds.AddLine(bounds.Left, bounds.Bottom - radius, bounds.Left, bounds.Top + radius);
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Left, bounds.Top, radius, radius, 180, 90);
                        // line top
                        pathBounds.AddLine(bounds.Left + radius, bounds.Top, bounds.Right - radius, bounds.Top);
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Right - radius, bounds.Top, radius, radius, 270, 90);
                        // line right
                        pathBounds.AddLine(bounds.Right, bounds.Top + radius, bounds.Right, bounds.Bottom - radius);
                        // pointed path //
                        if (inset == -1)
                            radius = 0;
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Right - radius, bounds.Bottom - radius, radius, radius, 0, 90);
                        // line bottom right
                        pathBounds.AddLine(bounds.Right - radius, bounds.Bottom, bounds.Right - (radius + inset), bounds.Bottom);
                        // pointed center
                        diff = (bounds.Width / 2) - ((int)radius + inset);
                        // right half
                        pathBounds.AddLine(bounds.Right - (radius + inset), bounds.Bottom, bounds.Right - (radius + inset + diff), bounds.Bottom + depth);
                        // left half
                        pathBounds.AddLine(bounds.Right - (radius + inset + diff), bounds.Bottom + depth, bounds.Left + radius + inset, bounds.Bottom);
                        // line bottom left
                        pathBounds.AddLine(bounds.Left + radius + inset, bounds.Bottom, bounds.Left + radius, bounds.Bottom);
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Left, bounds.Bottom - radius, radius, radius, 90, 90);

                        break;
                    }
                case PointDirection.Right:
                    {
                        // line top
                        pathBounds.AddLine(bounds.Left + radius, bounds.Top, bounds.Right - (radius + depth), bounds.Top);
                        // arc top right
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Right - (radius + depth), bounds.Top, radius, radius, 270, 90);

                        // top line
                        pathBounds.AddLine(bounds.Right - depth, bounds.Top + radius, bounds.Right - depth, bounds.Top + (radius + inset));
                        // pointed center
                        diff = (bounds.Height / 2) - ((int)radius + inset);
                        // top half
                        pathBounds.AddLine(bounds.Right - depth, bounds.Top + (radius + inset), bounds.Right, bounds.Bottom - (radius + inset + diff));
                        // bottom half
                        pathBounds.AddLine(bounds.Right, bounds.Bottom - (radius + inset + diff), bounds.Right - depth, bounds.Bottom - (radius + inset));
                        // line right bottom
                        pathBounds.AddLine(bounds.Right - depth, bounds.Bottom - (radius + inset), bounds.Right - depth, bounds.Bottom - radius);
                        // arc bottom right
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Right - (radius + depth), bounds.Bottom - radius, radius, radius, 0, 90);
                        // line bottom
                        pathBounds.AddLine(bounds.Right - (radius + depth), bounds.Bottom, bounds.Left + radius, bounds.Bottom);
                        if (inset == -1)
                            radius = 0;
                        // arc bottom left
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Left, bounds.Bottom - radius, radius, radius, 90, 90);
                        // line left
                        pathBounds.AddLine(bounds.Left, bounds.Bottom - radius, bounds.Left, bounds.Top + radius);
                        // arc top left
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Left, bounds.Top, radius, radius, 180, 90);

                        /*// pointed path //
                        // line left bottom
                        pathBounds.AddLine(bounds.Left, bounds.Bottom - radius, bounds.Left, bounds.Bottom - (radius + inset));
                        // pointed center
                        diff = (bounds.Height / 2) - ((int)radius + inset);
                        // bottom half
                        pathBounds.AddLine(bounds.Left, bounds.Bottom - (radius + inset), bounds.Left - depth, bounds.Bottom - (radius + inset + diff));
                        // top half
                        pathBounds.AddLine(bounds.Left - depth, bounds.Bottom - (radius + inset + diff), bounds.Left, bounds.Top + (radius + inset));
                        // top line
                        pathBounds.AddLine(bounds.Left, bounds.Top + (radius + inset), bounds.Left, bounds.Top + radius);
                        if (radius > 0)
                            pathBounds.AddArc(bounds.Left, bounds.Top, radius, radius, 180, 90);*/
                        break;
                    }
            }

            pathBounds.CloseFigure();
            return pathBounds;
        }

        /// <summary>Create a round GraphicsPath [not used]</summary>
        private GraphicsPath CreateRoundPath(Graphics g, Rectangle bounds)
        {
            int size = bounds.Width > bounds.Height ? bounds.Height : bounds.Width;
            bounds.Height = size;
            bounds.Width = size;
            GraphicsPath circlePath = new GraphicsPath();
            // create the path
            circlePath.AddEllipse(bounds);
            circlePath.CloseFigure();
            return circlePath;
        }

        /// <summary>Create a rounded rectangle GraphicsPath</summary>
        private GraphicsPath CreateRoundRectanglePath(Graphics g, Rectangle bounds, float radius)
        {
            // create a path
            GraphicsPath pathBounds = new GraphicsPath();
            // arc top left
            pathBounds.AddArc(bounds.Left, bounds.Top, radius, radius, 180, 90);
            // line top
            pathBounds.AddLine(bounds.Left + radius, bounds.Top, bounds.Right - radius, bounds.Top);
            // arc top right
            pathBounds.AddArc(bounds.Right - radius, bounds.Top, radius, radius, 270, 90);
            // line right
            pathBounds.AddLine(bounds.Right, bounds.Top + radius, bounds.Right, bounds.Bottom - radius);
            // arc bottom right
            pathBounds.AddArc(bounds.Right - radius, bounds.Bottom - radius, radius, radius, 0, 90);
            // line bottom
            pathBounds.AddLine(bounds.Right - radius, bounds.Bottom, bounds.Left + radius, bounds.Bottom);
            // arc bottom left
            pathBounds.AddArc(bounds.Left, bounds.Bottom - radius, radius, radius, 90, 90);
            // line left
            pathBounds.AddLine(bounds.Left, bounds.Bottom - radius, bounds.Left, bounds.Top + radius);
            pathBounds.CloseFigure();
            return pathBounds;
        }

        /// <summary>Set default button sizes</summary>
        private void DefaultButtonSize(ButtonType type)
        {
            Size sz = new Size(14, 14);
            switch (type)
            {
                case ButtonType.Hybrid:
                case ButtonType.Round:
                    _iTrackDepth = 4;
                    sz.Height = _iTrackDepth + 8;
                    sz.Width = sz.Height;
                    break;
                case ButtonType.PointerDownLeft:
                case ButtonType.PointerUpRight:
                    _iTrackDepth = 5;
                    sz.Height = _iTrackDepth * 4;
                    sz.Width = _iTrackDepth + 2;
                    break;
                case ButtonType.RoundedRectInline:
                    _iTrackDepth = 4;
                    _iButtonCornerRadius = 6;
                    sz.Height = _iTrackDepth + 6;
                    sz.Width = sz.Height * 2;
                    break;
                case ButtonType.RoundedRectOverlap:
                    _iTrackDepth = 4;
                    _iButtonCornerRadius = 4;
                    sz.Height = _iTrackDepth * 4;
                    sz.Width = _iTrackDepth + 6;
                    break;
                case ButtonType.GlassInline:
                    _iTrackDepth = 6;
                    _iButtonCornerRadius = 2;
                    sz.Height = _iTrackDepth + 6;
                    sz.Width = sz.Height * 2;
                    break;
                case ButtonType.GlassOverlap:
                    _iTrackDepth = 4;
                    _iButtonCornerRadius = 2;
                    sz.Height = _iTrackDepth * 4;
                    sz.Width = _iTrackDepth + 6;
                    break;
            }

            if (Orientation == Orientation.Vertical)
            {
                Size sv = new Size(sz.Height, sz.Width);
                _szButtonSize = sv;
            }
            else
            {
                _szButtonSize = sz;
            }
        }

        /// <summary>Return button size and coordinates</summary>
        private Rectangle GetButtonRectangle()
        {
            Rectangle bounds = new Rectangle(0, 0, this.Width, this.Height);
            RectangleF buttonRect = new RectangleF();
            int offsetX = (bounds.Width / 2);
            int offset = 0;
            double pos = _dButtonPosition + this.TrackPadding;

            if (Orientation == Orientation.Horizontal)
            {
                if (this.SliderFlyOut != FlyOutStyle.None)
                    offset = this.FlyOutMaxDepth + this.FlyOutSpacer;
                if (this.TickStyle == TickStyle.TopLeft)
                    offset += this.TickMaxLength + this.TickMinPadding;
                else if (this.TickStyle == TickStyle.BottomRight)
                    offset -= this.TickMaxLength + this.TickMinPadding;
                offset += (int)((bounds.Height - offset) * .5f);
                offset -= (int)(this.ButtonSize.Height * .5f);
                buttonRect = new RectangleF((float)pos, offset, this.ButtonSize.Width, this.ButtonSize.Height);
            }
            else
            {
                // offset on track
                if (this.SliderFlyOut != FlyOutStyle.None)
                    offset = this.FlyOutMaxDepth + this.FlyOutSpacer;
                if (this.TickStyle == TickStyle.TopLeft)
                    offset += this.TickMaxLength + this.TickMinPadding;
                else if (this.TickStyle == TickStyle.BottomRight)
                    offset -= this.TickMaxLength + this.TickMinPadding;
                offset += (int)((bounds.Width - offset) * .5f);
                offset -= (int)(this.ButtonSize.Width * .5f);
                buttonRect = new RectangleF(offset, (float)pos, this.ButtonSize.Width, this.ButtonSize.Height);
            }
            // store it for hit testing
            _buttonRect = new RECT((int)buttonRect.X, (int)buttonRect.Y, (int)buttonRect.Right, (int)buttonRect.Bottom);
            return Rectangle.Round(buttonRect);
        }

        /// <summary>Return track size and coordinates</summary>
        private Rectangle GetTrackRectangle()
        {
            Rectangle bounds = new Rectangle(0, 0, this.Width, this.Height);
            Rectangle trackRect;
            int offset;

            if (Orientation == Orientation.Horizontal)
            {
                // reduce for padding and center rect
                offset = (int)(this.TrackPadding);
                bounds.Inflate(-offset, 0);
                offset = 0;
                if (this.SliderFlyOut != FlyOutStyle.None)
                    offset = this.FlyOutMaxDepth + this.FlyOutSpacer;
                if (this.TickStyle == TickStyle.TopLeft)
                    offset += this.TickMaxLength + this.TickMinPadding;
                else if (this.TickStyle == TickStyle.BottomRight)
                    offset -= this.TickMaxLength + this.TickMinPadding;
                offset += (int)((bounds.Height - offset) * .5f);
                offset -= (int)(this.TrackDepth * .5f);
                trackRect = new Rectangle(bounds.X, offset, bounds.Width, this.TrackDepth);
            }
            else
            {
                offset = (int)(this.TrackPadding);
                bounds.Inflate(0, -offset);
                offset = 0;
                if (this.SliderFlyOut != FlyOutStyle.None)
                    offset = this.FlyOutMaxDepth + this.FlyOutSpacer;
                if (this.TickStyle == TickStyle.TopLeft)
                    offset += this.TickMaxLength + this.TickMinPadding;
                else if (this.TickStyle == TickStyle.BottomRight)
                    offset -= this.TickMaxLength + this.TickMinPadding;
                offset += (int)((bounds.Width - offset) * .5f);
                offset -= (int)(this.TrackDepth * .5f);
                trackRect = new Rectangle(offset, bounds.Y, this.TrackDepth, bounds.Height);
            }
            // store for hit testing
            _trackRect = new RECT(trackRect.X, trackRect.Y, trackRect.Right, trackRect.Bottom);
            return trackRect;
        }

        /// <summary>Exact size between ticks</summary>
        private double Increment()
        {
            Rectangle trackRect = GetTrackRectangle();

            if (Orientation == Orientation.Horizontal)
                return (double)((trackRect.Width - this.ButtonSize.Width) / IncrementScale());
            else
                return (double)((trackRect.Height - this.ButtonSize.Height) / IncrementScale());
        }

        /// <summary>Offset number if Minimum and Maximum are negative to positive integers</summary>
        private double IncrementOffset()
        {
            double center = 0;
            if (this.Minimum < 0 && this.Maximum > 0)
                center = 1;
            return center;
        }

        /// <summary>The total number of tick increments</summary>
        private double IncrementScale()
        {
            return (double)Math.Abs(this.Maximum - this.Minimum);
        }

        /// <summary>The incremental size between the Minimum and Value</summary>
        private double IncrementalValue()
        {
            if (Orientation == Orientation.Horizontal)
                return Increment() * (double)(Math.Abs(this.Value - this.Minimum));
            else
                return Increment() * (double)(IncrementScale() - (Math.Abs(this.Value - this.Minimum)));
        }

        /// <summary>Modulus returns true if divisible with no remainder</summary>
        private bool Mod(int a, int b)
        {
            if (Math.IEEERemainder((double)a, (double)b) == 0)
                return true;
            return false;
        }

        /// <summary>Return position coordinates from value</summary>
        private double PosFromValue(int val)
        {
            if (Orientation == Orientation.Horizontal)
                return Increment() * (double)(Math.Abs(val - this.Minimum + (val != this.Minimum ? IncrementOffset() : 0)));
            else
                return Increment() * (double)(IncrementScale() - (Math.Abs(val - this.Minimum) + (val != this.Minimum ? IncrementOffset() : 0)));
        }

        /// <summary>Repaint and optionally resize</summary>
        private void PropertyChange()
        {
            if (this.DesignMode)
            {
                if (this.SmoothScrolling)
                    _dButtonPosition = PosFromValue(this.Value);
                else
                    _dButtonPosition = IncrementalValue();
                if (this.AutoSize && this.FinishedPropRun)
                    ResizeThis();
                CreateGraphicsObjects();
            }
            else if (this.IsInited)
            {
                if (this.SmoothScrolling)
                    _dButtonPosition = PosFromValue(this.Value);
                else
                    _dButtonPosition = IncrementalValue();
                CreateGraphicsObjects();
            }
            DrawSlider();
        }

        /// <summary>Repaint the control</summary>
        private void Repaint()
        {
            this.Invalidate();
            this.Update();
        }

        /// <summary>Resize the control via alignments and options</summary>
        private void ResizeThis()
        {
            int offset = 0;
            int depth = 0;
            int diff = 0;
            Size sz = new Size(14, 14);

            CalculateMaximumSize();
            CalculateMinimumSize();

            if (this.Orientation == Orientation.Horizontal)
            {
                if (this.MinSize.Height == 0 || this.MaxSize.Height == 0)
                    return;

                if (this.Height <= this.MinSize.Height)
                    this.Height = this.MinSize.Height;
                else if (this.Height >= this.MaxSize.Height)
                    this.Height = this.MaxSize.Height;

                offset = this.Height - this.MinSize.Height;
                diff = this.MinSize.Height - ButtonSize.Height;

                if (offset < 2)
                    depth = TrackMinDepth;
                else if (offset < 3)
                    depth = TrackMinDepth + 1;
                else if (offset < 4)
                    depth = TrackMinDepth + 2;
                else
                    depth = TrackMaxDepth;

                switch (this.ButtonStyle)
                {
                    case ButtonType.GlassInline:
                        {
                            sz.Height = depth + 6;
                            sz.Width = sz.Height * 2;
                            break;
                        }
                    case ButtonType.RoundedRectInline:
                        {
                            sz.Height = depth + 6;
                            sz.Width = sz.Height * 2;
                            break;
                        }
                    case ButtonType.GlassOverlap:
                        {
                            sz.Height = depth * 4;
                            sz.Width = depth + 6;
                            break;
                        }
                    case ButtonType.RoundedRectOverlap:
                        {
                            sz.Height = depth * 4;
                            sz.Width = depth + 6;
                            break;
                        }
                    case ButtonType.Round:
                    case ButtonType.Hybrid:
                        {
                            sz.Height = depth + 8;
                            sz.Width = sz.Height;
                            break;
                        }
                    case ButtonType.PointerDownLeft:
                    case ButtonType.PointerUpRight:
                        {
                            sz.Height = depth * 4;
                            sz.Width = depth + 2;
                            break;
                        }
                    default:
                        {
                            sz.Height = depth * 4;
                            sz.Width = depth + 2;
                            break;
                        }
                }

                if (this.Width < (this.ButtonSize.Height + this.TrackPadding) * 2)
                    this.Width = (this.ButtonSize.Height + this.TrackPadding) * 2;
                if (sz.Height != this.ButtonSize.Height && this.Height >= sz.Height + diff)
                {
                    _iTrackDepth = depth;
                    _szButtonSize = sz;
                }
                else if (sz.Height != this.ButtonSize.Height && this.Height < this.ButtonSize.Height + diff)
                {
                    _iTrackDepth = depth;
                    _szButtonSize = sz;
                }
            }
            else
            {
                if (this.MinSize.Width == 0 || this.MaxSize.Width == 0)
                    return;

                if (this.Width <= this.MinSize.Width)
                    this.Width = this.MinSize.Width;
                else if (this.Width >= this.MaxSize.Width)
                    this.Width = this.MaxSize.Width;

                offset = this.Width - this.MinSize.Width;

                if (offset < 2)
                    depth = TrackMinDepth;
                else if (offset < 3)
                    depth = TrackMinDepth + 1;
                else if (offset < 4)
                    depth = TrackMinDepth + 2;
                else
                    depth = TrackMaxDepth;

                switch (this.ButtonStyle)
                {
                    case ButtonType.GlassInline:
                        {
                            sz.Width = depth + 6;
                            sz.Height = sz.Width * 2;
                            break;
                        }
                    case ButtonType.RoundedRectInline:
                        {
                            sz.Width = depth + 6;
                            sz.Height = sz.Width * 2;
                            break;
                        }
                    case ButtonType.GlassOverlap:
                        {
                            sz.Width = depth * 4;
                            sz.Height = depth + 6;
                            break;
                        }
                    case ButtonType.RoundedRectOverlap:
                        {
                            sz.Width = depth * 4;
                            sz.Height = depth + 6;
                            break;
                        }
                    case ButtonType.Round:
                    case ButtonType.Hybrid:
                        {
                            sz.Width = depth + 8;
                            sz.Height = sz.Width;
                            break;
                        }
                    case ButtonType.PointerDownLeft:
                    case ButtonType.PointerUpRight:
                        {
                            sz.Width = depth * 4;
                            sz.Height = depth + 2;
                            break;
                        }
                    default:
                        {
                            sz.Width = depth * 4;
                            sz.Height = depth + 2;
                            break;
                        }
                }

                if (this.Height < (this.ButtonSize.Width + this.TrackPadding) * 2)
                    this.Height = (this.ButtonSize.Width + this.TrackPadding) * 2;

                if (sz.Width != this.ButtonSize.Width && this.Width >= sz.Width + diff)
                {
                    _iTrackDepth = depth;
                    _szButtonSize = sz;
                }
                else if (sz.Width != this.ButtonSize.Width && this.Width < this.ButtonSize.Width + diff)
                {
                    _iTrackDepth = depth;
                    _szButtonSize = sz;
                }
            }
        }

        /// <summary>Scroll by large or small change values</summary>
        private void ScrollChange(ChangeType change, bool decrease)
        {
            int count = 0;
            if (change == ChangeType.Large)
            {
                if (decrease)
                    count = this.Value - LargeChange;
                else
                    count = this.Value + LargeChange;
            }
            else
            {
                if (decrease)
                    count = this.Value - SmallChange;
                else
                    count = this.Value + SmallChange;
            }

            if (count < this.Minimum)
                count = this.Minimum;
            if (count > this.Maximum)
                count = this.Maximum;

            this.Value = count;
        }

        /// <summary>Estimate value from position</summary>
        public int ValueFromPosition()
        {
            try
            {
                int pos = this.PointToClient(Cursor.Position).X;
                double increment = Increment();
                int sz = (Orientation == Orientation.Horizontal) ? this.ButtonSize.Width : this.ButtonSize.Height;
                double val = IncrementalValue();
                pos -= (sz / 2);

                if (pos > -sz)
                {
                    if (pos < this.TrackPadding)
                        pos = this.TrackPadding;
                    if (this.Orientation == Orientation.Horizontal)
                    {
                        if (pos > PosFromValue(this.Maximum) + this.TrackPadding)
                            pos = (int)PosFromValue(this.Maximum) + this.TrackPadding;
                    }
                    else
                    {
                        if (pos > PosFromValue(this.Minimum) + this.TrackPadding)
                            pos = (int)PosFromValue(this.Minimum) + this.TrackPadding;
                    }
                    pos -= this.TrackPadding;
                }
                return ValueFromPos(pos);
            }
            catch { return 0; }
        }

        /// <summary>Scroll the slider to a position and set value</summary>
        private void ScrollThis(double pos)
        {
            bool redraw = false;
            double val;
            double store = _dButtonPosition;
            double increment = Increment();
            int sz = (Orientation == Orientation.Horizontal) ? this.ButtonSize.Width : this.ButtonSize.Height;

            val = IncrementalValue();
            pos -= (sz / 2); //ju 1.3

            if (pos > -sz)
            {
                if (SmoothScrolling)
                {
                    if (pos < this.TrackPadding)
                        pos = this.TrackPadding;
                    if (this.Orientation == Orientation.Horizontal)
                    {
                        if (pos > PosFromValue(this.Maximum) + this.TrackPadding)
                            pos = PosFromValue(this.Maximum) + this.TrackPadding;
                    }
                    else
                    {
                        if (pos > PosFromValue(this.Minimum) + this.TrackPadding)
                            pos = PosFromValue(this.Minimum) + this.TrackPadding;
                    }
                    pos -= this.TrackPadding;
                    _dButtonPosition = pos;
                    if (store != pos)
                    {
                        val = this.Value;
                        _iValue = ValueFromPos(pos);
                        if (_iValue != val)
                        {
                            if (ValueChanged != null)
								ValueChanged(this, EventArgs.Empty);
                        }
                        DrawSlider();
                    }
					if (Scrolled != null)
						Scrolled(this, EventArgs.Empty);
				}
                else
                {
                    store = this.Value;
                    //pos -= this.TrackPadding; ju 1.3

                    if (pos > val + increment &&
                        (Orientation == Orientation.Horizontal && this.Value != this.Maximum) ||
                            (Orientation == Orientation.Vertical && this.Value != this.Minimum))
                    {
                        _iValue = ValueFromPos(pos);
                        if (Scrolled != null)
                            Scrolled(this, EventArgs.Empty);
                    }
                    else if (pos < val && // ju 1.3
                        (Orientation == Orientation.Horizontal && this.Value != this.Minimum) ||
                            (Orientation == Orientation.Vertical && this.Value != this.Maximum))
                    {
                        _iValue = ValueFromPos(pos);
                        if (Scrolled != null)
                            Scrolled(this, EventArgs.Empty);
                    }
                    if (_iValue != store)
                        this.Value = _iValue;
                    else if (redraw)
                        DrawSlider();
                }

            }
        }

        /// <summary>Mouse Hit test</summary>
        private HitTest SliderHitTest()
        {
            Point pt = new Point();
            RECT tr = new RECT();

            GetClientRect(this.Handle, ref tr);
            GetCursorPos(ref pt);
            ScreenToClient(this.Handle, ref pt);
            if (PtInRect(ref _buttonRect, pt))
                return HitTest.Button;
            else if (PtInRect(ref _trackRect, pt))
                return HitTest.Track;
            else
                return HitTest.Nowhere;
        }

        /// <summary>The value at a provided position</summary>
        private int ValueFromPos(double pos)
        {
            int val;
            //pos -= this.TrackPadding;
            if (Orientation == Orientation.Horizontal)
                val = this.Minimum + (int)Math.Round(pos / Increment());
            else
                val = this.Maximum - (int)Math.Round(pos / Increment());

            if (val < this.Minimum)
                val = this.Minimum;
            if (val > this.Maximum)
                val = this.Maximum;
            return val;
        }
        #endregion
        #endregion

        #region Graphics Mode
        /// <summary>Maintains graphic object state</summary>
        internal class GraphicsMode : IDisposable
        {
            #region Fields
            private Graphics _gGraphicCopy;
            private SmoothingMode _eOldMode;
            #endregion

            #region Methods
            /// <summary>
            /// Initialize a new instance of the class.
            /// </summary>
            /// <param name="g">Graphics instance.</param>
            /// <param name="mode">Desired Smoothing mode.</param>
            public GraphicsMode(Graphics g, SmoothingMode mode)
            {
                _gGraphicCopy = g;
                _eOldMode = _gGraphicCopy.SmoothingMode;
                _gGraphicCopy.SmoothingMode = mode;
            }

            /// <summary>
            /// Revert the SmoothingMode to original setting.
            /// </summary>
            public void Dispose()
            {
                _gGraphicCopy.SmoothingMode = _eOldMode;
            }
            #endregion
        }
        #endregion

        #region Effects Timer
        /// <summary>Effect timer class</summary>
        internal class FadeTimer : IDisposable
        {
            #region Enum
            internal enum FadeType
            {
                None = 0,
                FadeIn,
                FadeOut,
                FadeFast,
                Loop
            }
            #endregion

            #region Structs
            [StructLayout(LayoutKind.Sequential)]
            private struct RECT
            {
                public RECT(int X, int Y, int Width, int Height)
                {
                    this.Left = X;
                    this.Top = Y;
                    this.Right = Width;
                    this.Bottom = Height;
                }
                public int Left;
                public int Top;
                public int Right;
                public int Bottom;
            }
            #endregion

            #region API
            [DllImport("user32.dll")]
            private static extern IntPtr GetDC(IntPtr handle);

            [DllImport("user32.dll")]
            private static extern int ReleaseDC(IntPtr handle, IntPtr hdc);

            [DllImport("gdi32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            private static extern bool BitBlt(IntPtr hdc, int nXDest, int nYDest, int nWidth, int nHeight, IntPtr hdcSrc, int nXSrc, int nYSrc, int dwRop);

            [DllImport("user32.dll")]
            private static extern IntPtr GetDesktopWindow();

            [DllImport("user32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            private static extern bool GetWindowRect(IntPtr hWnd, ref RECT lpRect);
            #endregion

            #region Events
            public delegate void CompleteDelegate(object sender);
            public delegate void TickDelegate(object sender);
            public event CompleteDelegate Complete;
            public event TickDelegate Tick;
            #endregion

            #region Fields
            private bool _bCaptureScreen = false;
            private bool _bCancelTimer;
            private bool _bIsReset;
            private int _iTickCounter;
            private int _iTickMaximum;
            private double _iTickRate;
            private FadeType _eFadeType;
            private cStoreDc _cButtonDc;
            private UserControl _ctParentControl;
            private System.Timers.Timer _aTimer;
            private bool _bInvalidating = false;
            #endregion

            #region Constructor
            public FadeTimer(object sender)
            {
                _iTickCounter = 0;
                _iTickMaximum = 10;
                _ctParentControl = (UserControl)sender;
                _aTimer = new System.Timers.Timer();
                _iTickRate = _aTimer.Interval;
                _aTimer.SynchronizingObject = (ISynchronizeInvoke)sender;
                _aTimer.Elapsed += new ElapsedEventHandler(OnTimedEvent);
            }
            #endregion

            #region Properties
            public cStoreDc ButtonDc
            {
                get { return _cButtonDc; }
                set { _cButtonDc = value; }
            }

            public bool CaptureScreen
            {
                get { return _bCaptureScreen; }
                set { _bCaptureScreen = value; }
            }

            public bool Invalidating
            {
                get { return _bInvalidating; }
                set { _bInvalidating = value; }
            }

            public bool IsReset
            {
                get { return _bIsReset; }
                set { _bIsReset = value; }
            }

            public bool Cancel
            {
                get { return _bCancelTimer; }
                set { _bCancelTimer = value; }
            }

            public bool Enabled
            {
                get { return _aTimer.Enabled; }
            }

            public FadeType FadeStyle
            {
                get { return _eFadeType; }
                set { _eFadeType = value; }
            }

            public double Interval
            {
                get { return _iTickRate; }
                set
                {
                    _iTickRate = value;
                    _aTimer.Interval = _iTickRate;
                }
            }

            public int TickCount
            {
                get { return _iTickCounter; }
                set { _iTickCounter = value; }
            }

            public int TickMaximum
            {
                get { return _iTickMaximum; }
                set { _iTickMaximum = value; }
            }
            #endregion

            #region Public Methods
            public void Dispose()
            {
                Reset();
                if (_cButtonDc != null)
                    _cButtonDc.Dispose();
                if (_aTimer != null)
                    _aTimer.Dispose();
                GC.SuppressFinalize(this);
            }

            public void Fade(FadeType ft)
            {
                Cancel = false;
                IsReset = false;
                Invalidating = false;
                _eFadeType = ft;
                if (_eFadeType == FadeType.FadeIn)
                {
                    TickCount = 0;
                    if (CaptureScreen)
                        CaptureDc();
                }
                else if (_eFadeType == FadeType.FadeOut)
                {
                    TickCount = 10;
                }
                else if (_eFadeType == FadeType.FadeFast)
                {
                    TickCount = 10;
                }
                else if (_eFadeType == FadeType.Loop)
                {
                    TickMaximum = 100000;
                    TickCount = 0;
                    if (CaptureScreen)
                        CaptureDc();
                }
                _aTimer.Enabled = true;
            }

            public void Stop()
            {
                _aTimer.Stop();
            }

            public void Reset()
            {
                TickCount = 0;
                _eFadeType = FadeType.None;
                IsReset = true;
                _aTimer.Stop();
                _aTimer.Enabled = false;
            }
            #endregion

            #region Event Handlers
            private void OnTimedEvent(object source, ElapsedEventArgs e)
            {
                if (Cancel)
                {
                    Invalidating = true;
                    if (Complete != null) Complete(this);
                    return;
                }
                else
                {
                    switch (_eFadeType)
                    {
                        case FadeType.FadeIn:
                            FadeIn();
                            break;
                        case FadeType.FadeFast:
                            FadeOut();
                            break;
                        case FadeType.FadeOut:
                            FadeOut();
                            break;
                        case FadeType.Loop:
                            FadeLoop();
                            break;
                    }
                }
            }
            #endregion

            #region private Methods
            private void CaptureDc()
            {
                try
                {
                    _cButtonDc.Width = _ctParentControl.Width;
                    _cButtonDc.Height = _ctParentControl.Height;
                    if (_cButtonDc.Hdc != IntPtr.Zero)
                    {
                        using (Graphics g = Graphics.FromHdc(_cButtonDc.Hdc))
                        {
                            RECT boundedRect = new RECT();
                            GetWindowRect(_ctParentControl.Handle, ref boundedRect);
                            g.CopyFromScreen(boundedRect.Left, boundedRect.Top, 0, 0, new Size(_cButtonDc.Width, _cButtonDc.Height), CopyPixelOperation.SourceCopy);
                        }
                    }
                }
                catch { }
            }

            private void FadeIn()
            {
                if (TickCount < TickMaximum)
                {
                    TickCount++;
                    if (Tick != null)
                        Tick(this);
                }
                else
                {
                    TickCount = TickMaximum;
                }
            }

            private void FadeLoop()
            {
                if (TickCount < TickMaximum)
                {
                    TickCount++;
                    if (Tick != null)
                        Tick(this);
                }
                else
                {
                    TickCount = TickMaximum;
                    Reset();
                    Invalidating = true;
                    if (Complete != null)
                        Complete(this);
                }
            }

            private void FadeOut()
            {
                if (TickCount > 0)
                {
                    if (_eFadeType == FadeType.FadeFast)
                    {
                        TickCount -= 2;
                        if (TickCount < 0)
                            TickCount = 0;
                    }
                    else
                    {
                        TickCount--;
                    }
                    if (Tick != null)
                        Tick(this);
                }
                else
                {
                    Reset();
                    Invalidating = true;
                    if (Complete != null)
                        Complete(this);
                }
            }

            ~FadeTimer()
            {
                Dispose();
            }
            #endregion
        }
        #endregion

		public class FlyOutEventArgs : EventArgs
		{
			public string text = "";
		}

        #region StoreDc
        /// <summary>DC buffer class</summary>
        internal class cStoreDc
        {
            #region API
            [DllImport("gdi32.dll")]
            private static extern IntPtr CreateDCA([MarshalAs(UnmanagedType.LPStr)]string lpszDriver, [MarshalAs(UnmanagedType.LPStr)]string lpszDevice, [MarshalAs(UnmanagedType.LPStr)]string lpszOutput, int lpInitData);

            [DllImport("gdi32.dll")]
            private static extern IntPtr CreateDCW([MarshalAs(UnmanagedType.LPWStr)]string lpszDriver, [MarshalAs(UnmanagedType.LPWStr)]string lpszDevice, [MarshalAs(UnmanagedType.LPWStr)]string lpszOutput, int lpInitData);

            [DllImport("gdi32.dll")]
            private static extern IntPtr CreateDC(string lpszDriver, string lpszDevice, string lpszOutput, int lpInitData);

            [DllImport("gdi32.dll")]
            private static extern IntPtr CreateCompatibleDC(IntPtr hdc);

            [DllImport("gdi32.dll")]
            private static extern IntPtr CreateCompatibleBitmap(IntPtr hdc, int nWidth, int nHeight);

            [DllImport("gdi32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            private static extern bool DeleteDC(IntPtr hdc);

            [DllImport("gdi32.dll", ExactSpelling = true, PreserveSig = true)]
            private static extern IntPtr SelectObject(IntPtr hdc, IntPtr hgdiobj);

            [DllImport("gdi32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            private static extern bool DeleteObject(IntPtr hObject);
            #endregion

            #region Fields
            private int _Height = 0;
            private int _Width = 0;
            private IntPtr _Hdc = IntPtr.Zero;
            private IntPtr _Bmp = IntPtr.Zero;
            private IntPtr _BmpOld = IntPtr.Zero;
            #endregion

            #region Methods
            public IntPtr Hdc
            {
                get { return _Hdc; }
            }

            public IntPtr HBmp
            {
                get { return _Bmp; }
            }

            public int Height
            {
                get { return _Height; }
                set
                {
                    if (_Height != value)
                    {
                        _Height = value;
                        ImageCreate(_Width, _Height);
                    }
                }
            }

            public int Width
            {
                get { return _Width; }
                set
                {
                    if (_Width != value)
                    {
                        _Width = value;
                        ImageCreate(_Width, _Height);
                    }
                }
            }

            public void SelectImage(Bitmap image)
            {
                if (Hdc != IntPtr.Zero && image != null)
                    SelectObject(Hdc, image.GetHbitmap());
            }

            private void ImageCreate(int Width, int Height)
            {
                IntPtr pHdc = IntPtr.Zero;

                ImageDestroy();
                pHdc = CreateDCA("DISPLAY", "", "", 0);
                _Hdc = CreateCompatibleDC(pHdc);
                _Bmp = CreateCompatibleBitmap(pHdc, _Width, _Height);
                _BmpOld = SelectObject(_Hdc, _Bmp);
                if (_BmpOld == IntPtr.Zero)
                {
                    ImageDestroy();
                }
                else
                {
                    _Width = Width;
                    _Height = Height;
                }
                DeleteDC(pHdc);
                pHdc = IntPtr.Zero;
            }

            private void ImageDestroy()
            {
                if (_BmpOld != IntPtr.Zero)
                {
                    SelectObject(_Hdc, _BmpOld);
                    _BmpOld = IntPtr.Zero;
                }
                if (_Bmp != IntPtr.Zero)
                {
                    DeleteObject(_Bmp);
                    _Bmp = IntPtr.Zero;
                }
                if (_Hdc != IntPtr.Zero)
                {
                    DeleteDC(_Hdc);
                    _Hdc = IntPtr.Zero;
                }
            }

            public void Dispose()
            {
                ImageDestroy();
            }
            #endregion
        }
        #endregion
    }
}

