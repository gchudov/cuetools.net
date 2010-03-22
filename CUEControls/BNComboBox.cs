using System.Windows.Forms;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms.Design;
using System.ComponentModel;
using System;

namespace BBBNOVA
{
    public class BNComboBox : ListControl
    {
        #region Variables

        private bool hovered = false;
        private bool resize = false;

        private Color _backColor = Color.White;
        private Color _color1 = Color.White;
        private Color _color2 = Color.Gainsboro;
        private Color _color3 = Color.White;
        private Color _color4 = Color.PaleGoldenrod;
        private BNRadius _radius = new BNRadius();

        private int _dropDownHeight = 200;
        private int _dropDownWidth = 0;
        private int _maxDropDownItems = 8;
        
        //private int _selectedIndex = -1;

        private bool _isDroppedDown = false;

        private ComboBoxStyle _dropDownStyle = ComboBoxStyle.DropDownList;

        private Rectangle rectBtn = new Rectangle(0, 0, 1, 1);
        private Rectangle rectContent = new Rectangle(0, 0, 1, 1);

        private ToolStripControlHost _controlHost;
        private ListBox _listBox;
        private ToolStripDropDown _popupControl;
        private TextBox _textBox;

		private DrawMode _drawMode = DrawMode.Normal;

        #endregion

        #region Delegates

        [Category("Behavior"), Description("Occurs when IsDroppedDown changed to True.")]
        public event EventHandler DroppedDown;

        [Category("Behavior"), Description("Occurs when the SelectedIndex property changes.")]
        public event EventHandler SelectedIndexChanged;

		//[Category("Behavior"), Description("Occurs when the SelectedValue property changes.")]
		//public event EventHandler SelectedValueChanged;

        [Category("Behavior"), Description("Occurs whenever a particular item/area needs to be painted.")]
		public event EventHandler<DrawItemEventArgs> DrawItem;

        [Category("Behavior"), Description("Occurs whenever a particular item's height needs to be calculated.")]
        public event EventHandler<MeasureItemEventArgs> MeasureItem;

        #endregion



        
        #region Properties

        public Color Color1
        {
            get { return _color1; }
            set { _color1 = value; Invalidate(true); }
        }

        public Color Color2
        {
            get { return _color2; }
            set { _color2 = value; Invalidate(true); }
        }
        
        public Color Color3
        {
            get { return _color3; }
            set { _color3 = value; Invalidate(true); }
        }

        public Color Color4
        {
            get { return _color4; }
            set { _color4 = value; Invalidate(true); }
        }

        public int DropDownHeight
        {
            get { return _dropDownHeight; }
            set { _dropDownHeight = value; }
        }

        public ListBox.ObjectCollection Items
        {
            get { return _listBox.Items; }
        }

        public int DropDownWidth
        {
            get { return _dropDownWidth; }
            set { _dropDownWidth = value; }
        }

        public int MaxDropDownItems
        {
            get { return _maxDropDownItems; }
            set { _maxDropDownItems = value; }
        }

        public new object DataSource
        {
            get { return base.DataSource; }
            set 
            { 
                _listBox.DataSource = value;
                base.DataSource = value;
                OnDataSourceChanged(System.EventArgs.Empty);
            }
        }

        public bool Sorted
        {
            get
            {
                return _listBox.Sorted;
            }
            set
            {
                _listBox.Sorted = value;
            }
        }

		[DefaultValue(DrawMode.Normal)]
        [Category("Behavior"), Description("Indicates whether the user code or this control will handle the drawing of elements in the list.")]
        public DrawMode DrawMode
        {
			get { return _drawMode; }
            set
            {
				_drawMode = value;
                _listBox.DrawMode = value == DrawMode.Normal ? DrawMode.OwnerDrawFixed : value;
            }
        }
        
        public ComboBoxStyle DropDownStyle
        {
            get { return _dropDownStyle; }
            set 
            { 
                _dropDownStyle = value; 
            
                if (_dropDownStyle == ComboBoxStyle.DropDownList)
                {
                    _textBox.Visible = false;
                }
                else
                {
                    _textBox.Visible = true;
                }
                Invalidate(true);
            }
        }

        public new Color BackColor
        {
            get { return _backColor; }
            set 
            { 
                this._backColor = value;
                _textBox.BackColor = value;
                Invalidate(true);
            }
        }

        public bool IsDroppedDown
        {
            get { return _isDroppedDown; }
            set 
            {
                if (_isDroppedDown == true && value == false )
                {
                    if (_popupControl.IsDropDown)
                    {
                        _popupControl.Close();
                    }
                }

                _isDroppedDown = value;

                if (_isDroppedDown)
                {
					_listBox.Width = _dropDownWidth;
					_listBox.Height = CalculateListHeight();
                    _popupControl.Show(this, CalculateDropPosition(), ToolStripDropDownDirection.BelowRight);
					Capture = false;
					_listBox.Capture = true;
				}

                Invalidate();
                if (_isDroppedDown)
                    OnDroppedDown(this, EventArgs.Empty);
            }
        }

		[Category("Appearance"), Description("Selects the radius of combobox edges.")]
        public BNRadius Radius
        {
            get { return _radius; }
			set { _radius = value; }
        }

		[Category("Appearance"), Description("Selects the type of border around drop down list."), DefaultValue(BorderStyle.None)]
		public BorderStyle Border
		{
			get
			{
				return _listBox.BorderStyle;
			}
			set
			{
				_listBox.BorderStyle = value;
			}
		}

		[Category("Appearance"), Description("Indicates whether a three-dimentional shadow effect appears when drop down list is activated."), DefaultValue(true)]
		public bool DropShadowEnabled
		{
			get
			{
				return _popupControl.DropShadowEnabled;
			}
			set
			{
				_popupControl.DropShadowEnabled = value;
			}
		}

		private ImageList _imageList = null;

		public ImageList ImageList
		{
			get
			{
				return _imageList;
			}
			set
			{
				_imageList = value;
				AdjustControls();
			}
		}

		private string _imageKeyMember = null;

		[Category("Data"), Description("Indicates the property to use as a key to select images from ImageList."), DefaultValue(null)]
		[TypeConverter("System.Windows.Forms.Design.DataMemberFieldConverter, System.Design, Version=2.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a")]
		[Editor("System.Windows.Forms.Design.DataMemberFieldEditor, System.Design, Version=2.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a", typeof(System.Drawing.Design.UITypeEditor))]
		[Browsable(true)]
		public string ImageKeyMember
		{
			get
			{				
				return _imageKeyMember;
			}
			set
			{
				_imageKeyMember = value;
			}
		}

        #endregion




        #region Constructor
        public BNComboBox()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint, true);
            SetStyle(ControlStyles.ContainerControl, true);
            SetStyle(ControlStyles.OptimizedDoubleBuffer, true);
            SetStyle(ControlStyles.ResizeRedraw, true);
            SetStyle(ControlStyles.Selectable, true);
            SetStyle(ControlStyles.SupportsTransparentBackColor, true);
            SetStyle(ControlStyles.UserMouse, true);
            SetStyle(ControlStyles.UserPaint, true);
            SetStyle(ControlStyles.Selectable, true);

            base.BackColor = Color.Transparent;
            _radius.BottomLeft = 2;
            _radius.BottomRight = 2;
            _radius.TopLeft = 2;
            _radius.TopRight = 6;

            this.Height = 21;
            this.Width = 95;

            this.SuspendLayout();
            _textBox = new TextBox();
            _textBox.BorderStyle = System.Windows.Forms.BorderStyle.None;
            _textBox.Location = new System.Drawing.Point(3, 4);
            _textBox.Size = new System.Drawing.Size(60, 13);
            _textBox.TabIndex = 0;
            _textBox.WordWrap = false;
            _textBox.Margin = new Padding(0);
            _textBox.Padding = new Padding(0);
            _textBox.TextAlign = HorizontalAlignment.Left;
			_textBox.Font = base.Font;
            this.Controls.Add(_textBox);
            this.ResumeLayout(false);

            AdjustControls();

            _listBox = new ListBox();
            _listBox.IntegralHeight = true;
            _listBox.BorderStyle = BorderStyle.FixedSingle;
            _listBox.SelectionMode = SelectionMode.One;
			_listBox.DrawMode = DrawMode.OwnerDrawFixed;
			_listBox.Font = base.Font;
			//_listBox.DrawMode = DrawMode.Normal;
            _listBox.BindingContext = new BindingContext();

            _controlHost = new ToolStripControlHost(_listBox);
            _controlHost.Padding = new Padding(0);
            _controlHost.Margin = new Padding(0);
            _controlHost.AutoSize = false;

            _popupControl = new ToolStripDropDown();
            _popupControl.Padding = new Padding(0);
            _popupControl.Margin = new Padding(0);
            _popupControl.AutoSize = true;
			_popupControl.AutoClose = false;
            _popupControl.DropShadowEnabled = true;
            _popupControl.Items.Add(_controlHost);

            _dropDownWidth = this.Width;

			_listBox.SelectedValueChanged += new EventHandler(_listBox_SelectedValueChanged);
			_listBox.SelectedIndexChanged += new EventHandler(_listBox_SelectedIndexChanged);
            _listBox.MeasureItem += new MeasureItemEventHandler(_listBox_MeasureItem);
            _listBox.DrawItem += new DrawItemEventHandler(_listBox_DrawItem);
            _listBox.MouseClick += new MouseEventHandler(_listBox_MouseClick);
			_listBox.MouseUp += new MouseEventHandler(_listBox_MouseUp);
			_listBox.MouseDown += new MouseEventHandler(_listBox_MouseDown);
            _listBox.MouseMove += new MouseEventHandler(_listBox_MouseMove);
			//(_listBox.DataManager as CurrencyManager).ListChanged += new ListChangedEventHandler(BNComboBox_ListChanged);

            _popupControl.Closed += new ToolStripDropDownClosedEventHandler(_popupControl_Closed);

            _textBox.Resize += new EventHandler(_textBox_Resize);
            _textBox.TextChanged += new EventHandler(_textBox_TextChanged);
		}

        #endregion




        #region Overrides

		protected override void OnCreateControl()
		{
			base.OnCreateControl();
			_listBox.CreateControl();
			_listBox.SelectedIndex = -1;
		}

        protected override void OnDataSourceChanged(EventArgs e)
        {
			_listBox.SelectedIndex = -1;
            base.OnDataSourceChanged(e);
		}

        protected override void OnDisplayMemberChanged(EventArgs e)
        {
            _listBox.DisplayMember = this.DisplayMember;
			_listBox.SelectedIndex = _listBox.SelectedIndex;
            //this.SelectedIndex = this.SelectedIndex;
            base.OnDisplayMemberChanged(e);
        }

        protected override void OnEnabledChanged(EventArgs e)
        {
            Invalidate(true);
            base.OnEnabledChanged(e);
        }

        protected override void OnForeColorChanged(EventArgs e)
        {
            _textBox.ForeColor = this.ForeColor;
            base.OnForeColorChanged(e);
        }

        protected override void OnFormatInfoChanged(EventArgs e)
        {
            _listBox.FormatInfo = this.FormatInfo;
            base.OnFormatInfoChanged(e);
        }

        protected override void OnFormatStringChanged(EventArgs e)
        {
            _listBox.FormatString = this.FormatString;
            base.OnFormatStringChanged(e);
        }

        protected override void OnFormattingEnabledChanged(EventArgs e)
        {
            _listBox.FormattingEnabled = this.FormattingEnabled;
            base.OnFormattingEnabledChanged(e);
        }

        public override Font Font
        {
            get
            {
                return base.Font;
            }
            set
            {
                resize = true;
                _textBox.Font = value;
				_listBox.Font = value;
                base.Font = value;
                Invalidate(true);
            }
        }

        protected override void OnControlAdded(ControlEventArgs e)
        {
            e.Control.MouseDown += new MouseEventHandler(Control_MouseDown);
            e.Control.MouseEnter += new EventHandler(Control_MouseEnter);
            e.Control.MouseLeave += new EventHandler(Control_MouseLeave);
            e.Control.GotFocus += new EventHandler(Control_GotFocus);
            e.Control.LostFocus += new EventHandler(Control_LostFocus);
            base.OnControlAdded(e);
        }        

        protected override void OnMouseEnter(EventArgs e)
        {
            hovered = true;
            this.Invalidate(true);
            base.OnMouseEnter(e);
        }

        protected override void OnMouseLeave(EventArgs e)
        {
            if (!this.RectangleToScreen(this.ClientRectangle).Contains(MousePosition))
            {
                hovered = false;
                Invalidate(true);
            }

            base.OnMouseLeave(e);
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
			//System.Diagnostics.Trace.WriteLine(string.Format("OnMouseDown({0})", SelectedIndex)); 
			_textBox.Focus();
			if (e.Button == MouseButtons.Left)
				if ((this.RectangleToScreen(rectBtn).Contains(MousePosition) || (DropDownStyle == ComboBoxStyle.DropDownList)))
				{
					this.IsDroppedDown = !this.IsDroppedDown;
				}
			base.OnMouseDown(e);
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
			hovered = this.RectangleToScreen(this.ClientRectangle).Contains(MousePosition);
            Invalidate();
        }

        protected override void OnMouseWheel(MouseEventArgs e)
        {
			if (e.Delta < 0 && _listBox.SelectedIndex < _listBox.Items.Count - 1)
				_listBox.SelectedIndex = _listBox.SelectedIndex + 1;
            else if (e.Delta > 0 && _listBox.SelectedIndex > 0)
				_listBox.SelectedIndex = _listBox.SelectedIndex - 1;

            base.OnMouseWheel(e);
        }

        protected override void OnGotFocus(EventArgs e)
        {
            Invalidate(true);
            base.OnGotFocus(e);
        }

		protected override void OnLostFocus(EventArgs e)
		{
			if (this.IsDroppedDown && !_popupControl.ContainsFocus && !this.ContainsFocus)
				this.IsDroppedDown = false;

			base.OnLostFocus(e);
		}

		private int _selectedIndex = -2;

		protected override void OnSelectedValueChanged(EventArgs e)
		{
			//System.Diagnostics.Trace.WriteLine(string.Format("OnSelectedValueChanged({0}=>{1})", Text, _listBox.Text));

			//this.SelectedIndex = _listBox.SelectedIndex;
			//this.Invalidate(true);
			Text = _listBox.SelectedValue != null ?
				_listBox.GetItemText(_listBox.SelectedValue) :
				"";

			OnSelectedIndexChanged(e);

			base.OnSelectedValueChanged(e);
		}

        protected override void OnSelectedIndexChanged(EventArgs e)
        {
			//System.Diagnostics.Trace.WriteLine(string.Format("OnSelectedIndexChanged({0}=>{1})", _selectedIndex, _listBox.SelectedIndex));
			if (_selectedIndex == _listBox.SelectedIndex)
				return;
			
			_selectedIndex = _listBox.SelectedIndex;

            if(SelectedIndexChanged!=null)
                SelectedIndexChanged(this, e);

            base.OnSelectedIndexChanged(e);
        }

        protected override void OnValueMemberChanged(EventArgs e)
        {
            _listBox.ValueMember = this.ValueMember;
			//_listBox.SelectedIndex = _listBox.SelectedIndex;
            base.OnValueMemberChanged(e);
        }

        protected override void OnResize(EventArgs e)
        {
            if (resize)
            {

                resize = false;
                AdjustControls();

                Invalidate(true);
            }
            else
                Invalidate(true);

            if (DesignMode)
                _dropDownWidth = this.Width;
        }

        public override string Text
        {
            get
            {
                return _textBox.Text;
            }
            set
            {
                _textBox.Text = value;
                base.Text = _textBox.Text;
				Invalidate(true);
                OnTextChanged(EventArgs.Empty);
            }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;

            //content border
            Rectangle rectCont = rectContent;
            rectCont.X += 1;
            rectCont.Y += 1;
            rectCont.Width -= 3;
            rectCont.Height -= 3;
            GraphicsPath pathContentBorder = CreateRoundRectangle(rectCont, Radius.TopLeft, Radius.TopRight, Radius.BottomRight,
                Radius.BottomLeft);

            //button border
            Rectangle rectButton = rectBtn;
            rectButton.X += 1;
            rectButton.Y += 1;
            rectButton.Width -= 3;
            rectButton.Height -= 3;
            GraphicsPath pathBtnBorder = CreateRoundRectangle(rectButton, 0, Radius.TopRight, Radius.BottomRight, 0);

            //outer border
            Rectangle rectOuter = rectContent;
            rectOuter.Width -= 1;
            rectOuter.Height -= 1;
            GraphicsPath pathOuterBorder = CreateRoundRectangle(rectOuter, Radius.TopLeft, Radius.TopRight, Radius.BottomRight,
                Radius.BottomLeft);

            //inner border
            Rectangle rectInner = rectContent;
            rectInner.X += 1;
            rectInner.Y += 1;
            rectInner.Width -= 3;
            rectInner.Height -= 3;
            GraphicsPath pathInnerBorder = CreateRoundRectangle(rectInner, Radius.TopLeft, Radius.TopRight, Radius.BottomRight,
                Radius.BottomLeft);

            //brushes and pens
            Brush brInnerBrush = new LinearGradientBrush(
                new Rectangle(rectInner.X,rectInner.Y,rectInner.Width,rectInner.Height+1), 
                (hovered || IsDroppedDown || ContainsFocus)?Color4:Color2, Color.Transparent,
                LinearGradientMode.Vertical);
            Brush brBackground;
            if (this.DropDownStyle == ComboBoxStyle.DropDownList)
            {
                brBackground = new LinearGradientBrush(pathInnerBorder.GetBounds(), 
                    Color.FromArgb(IsDroppedDown ? 100 : 255, Color.White), 
                    Color.FromArgb(IsDroppedDown?255:100, BackColor),
                    LinearGradientMode.Vertical);
            }
            else
            {
                brBackground = new SolidBrush(BackColor);
            }
            Pen penOuterBorder = new Pen(Color1, 0);
            Pen penInnerBorder = new Pen(brInnerBrush, 0);
            LinearGradientBrush brButtonLeft = new LinearGradientBrush(rectBtn, Color1, Color2, LinearGradientMode.Vertical);
            ColorBlend blend = new ColorBlend();
            blend.Colors = new Color[] { Color.Transparent, Color2, Color.Transparent };
            blend.Positions = new float[] { 0.0f, 0.5f, 1.0f};
            brButtonLeft.InterpolationColors = blend;
            Pen penLeftButton = new Pen(brButtonLeft, 0);
            Brush brButton = new LinearGradientBrush(pathBtnBorder.GetBounds(),
                Color.FromArgb(100, IsDroppedDown? Color2:Color.White),
                    Color.FromArgb(100, IsDroppedDown ? Color.White : Color2),
                    LinearGradientMode.Vertical);

            //draw
            e.Graphics.FillPath(brBackground, pathContentBorder);
            if (DropDownStyle != ComboBoxStyle.DropDownList)
            {
                e.Graphics.FillPath(brButton, pathBtnBorder);
            }
            e.Graphics.DrawPath(penOuterBorder, pathOuterBorder);
            e.Graphics.DrawPath(penInnerBorder, pathInnerBorder);

            e.Graphics.DrawLine(penLeftButton, rectBtn.Left + 1, rectInner.Top+1, rectBtn.Left + 1, rectInner.Bottom-1);
            

            //Glimph
            Rectangle rectGlimph = rectButton;
            rectButton.Width -= 4;
            e.Graphics.TranslateTransform(rectGlimph.Left + rectGlimph.Width / 2.0f, rectGlimph.Top + rectGlimph.Height / 2.0f);
            GraphicsPath path = new GraphicsPath();
            PointF[] points = new PointF[3];
            points[0] = new PointF(-6 / 2.0f, -3 / 2.0f);
            points[1] = new PointF(6 / 2.0f, -3 / 2.0f);
            points[2] = new PointF(0, 6 / 2.0f);
            path.AddLine(points[0], points[1]);
            path.AddLine(points[1], points[2]);
            path.CloseFigure();
            e.Graphics.RotateTransform(0);

            SolidBrush br = new SolidBrush(Enabled?Color.Gray:Color.Gainsboro);
            e.Graphics.FillPath(br, path);
            e.Graphics.ResetTransform();
            br.Dispose();
            path.Dispose();

			// image
			if (ImageList != null)
			{
				int idx = GetImageKey(SelectedIndex);
				if (idx >= 0)
					this.ImageList.Draw(e.Graphics, _textBox.Bounds.Left - this.ImageList.ImageSize.Width - 5, rectContent.Y + 2, idx);
			}

            //text
            if (DropDownStyle == ComboBoxStyle.DropDownList)
            {
                StringFormat sf  = new StringFormat(StringFormatFlags.NoWrap);
                sf.Alignment = StringAlignment.Near;

                Rectangle rectText = _textBox.Bounds;
                rectText.Offset(-3, 0);

                SolidBrush foreBrush = new SolidBrush(ForeColor);
                if (Enabled)
                {
                    e.Graphics.DrawString(_textBox.Text, this.Font, foreBrush, rectText.Location);
                }
                else
                {
                    ControlPaint.DrawStringDisabled(e.Graphics, _textBox.Text, Font, Color1, rectText, sf);
                }
            }
            /*
            Dim foreBrush As SolidBrush = New SolidBrush(color)
            If (enabled) Then
                g.DrawString(text, font, foreBrush, rect, sf)
            Else
                ControlPaint.DrawStringDisabled(g, text, font, backColor, _
                     rect, sf)
            End If
            foreBrush.Dispose()*/


            pathContentBorder.Dispose();
            pathOuterBorder.Dispose();
            pathInnerBorder.Dispose();
            pathBtnBorder.Dispose();

            penOuterBorder.Dispose();
            penInnerBorder.Dispose();
            penLeftButton.Dispose();

            brBackground.Dispose();
            brInnerBrush.Dispose();
            brButtonLeft.Dispose();
            brButton.Dispose();
        }

        #endregion




        #region ListControlOverrides

        public override int SelectedIndex
        {
            get { return _listBox.SelectedIndex; }
			set
			{
				_listBox.SelectedIndex = value;

				//_selectedIndex = value;
				//if (this.DataSource == null || value >= 0)
					//OnSelectedIndexChanged(EventArgs.Empty);
			}
		}

        public object SelectedItem
        {
            get { return _listBox.SelectedItem;  }
            set 
            { 
                _listBox.SelectedItem = value;
				//this.SelectedIndex = _listBox.SelectedIndex;
            }
        }

        public new object SelectedValue
        {
            get { return _listBox.SelectedValue; }
            set
            {
                _listBox.SelectedValue = value;
			}
        }

        protected override void RefreshItem(int index)
        {
            //throw new Exception("The method or operation is not implemented.");
        }

        protected override void RefreshItems()
        {
            //base.RefreshItems();
        }

        protected override void SetItemCore(int index, object value)
        {
            //base.SetItemCore(index, value);
        }

        protected override void SetItemsCore(System.Collections.IList items)
        {
            //throw new Exception("The method or operation is not implemented.");
        }

        #endregion




        #region NestedControlsEvents

        void Control_LostFocus(object sender, EventArgs e)
        {
            OnLostFocus(e);
        }

        void Control_GotFocus(object sender, EventArgs e)
        {
            OnGotFocus(e);
        }

        void Control_MouseLeave(object sender, EventArgs e)
        {
            OnMouseLeave(e);
        }

        void Control_MouseEnter(object sender, EventArgs e)
        {
            OnMouseEnter(e);
        }

        void Control_MouseDown(object sender, MouseEventArgs e)
        {
            OnMouseDown(e);
        }


		private int _hoverItem = -1;

        void _listBox_MouseMove(object sender, MouseEventArgs e)
        {
			int i = _listBox.IndexFromPoint(e.Location);
			if (_hoverItem != i)
			{
				_hoverItem = i;
				_listBox.Invalidate();
			}
        }

		void _listBox_MouseDown(object sender, MouseEventArgs e)
		{
			int i = _listBox.IndexFromPoint(e.Location);
			//System.Diagnostics.Trace.WriteLine(string.Format("_listBox_MouseDown({0})", i));
			if (i >= 0)
				_listBox.SelectedIndex = i;
			IsDroppedDown = false;
		}

		void _listBox_MouseUp(object sender, MouseEventArgs e)
		{
			int i = _listBox.IndexFromPoint(e.Location);
			//System.Diagnostics.Trace.WriteLine(string.Format("_listBox_MouseUp({0})", i));
			if (i >= 0)
			{
				_listBox.SelectedIndex = i;
				IsDroppedDown = false;
			}
		}

		void _listBox_MouseClick(object sender, MouseEventArgs e)
        {
			int i = _listBox.IndexFromPoint(e.Location);
			if (i >= 0)
			{
				_listBox.SelectedIndex = i;
				IsDroppedDown = false;
			}
        }

		private int GetImageKey(int index)
		{
			if (this.ImageList == null || index < 0)
				return -1;
			object key = FilterItemOnProperty(_listBox.Items[index], ImageKeyMember ?? DisplayMember);
			if (key == null)
				return -1;
			if (key is int)
				return (int)key;
			if (key is string)
				return this.ImageList.Images.IndexOfKey(key as string);
			return -1;
		}

		//void BNComboBox_ListChanged(object sender, ListChangedEventArgs e)
		//{
		//    _listBox.Height = CalculateListHeight();
		//}

        void _listBox_DrawItem(object sender, DrawItemEventArgs e)
        {
			if (DrawMode == DrawMode.Normal)
			{
				Color fg = _hoverItem != -1 && _hoverItem == e.Index ? Color1 : Color4;
				Color bg = _hoverItem != -1 && _hoverItem == e.Index ? Color4 : Color1;
				e.Graphics.FillRectangle(new SolidBrush(bg), e.Bounds);

				if (e.Index >= 0)
				{
					Rectangle textBounds = e.Bounds;

					textBounds.X += 5;
					textBounds.Width -= 5;

					// image
					if (this.ImageList != null)
					{
						int idx = GetImageKey(e.Index);
						if (idx >= 0)
							this.ImageList.Draw(e.Graphics, e.Bounds.X + 5, e.Bounds.Y, idx);
						textBounds.X += this.ImageList.ImageSize.Width + 5;
						textBounds.Width -= this.ImageList.ImageSize.Width + 5;
					}

					//text
					StringFormat sf = new StringFormat(StringFormatFlags.NoWrap);
					sf.Alignment = StringAlignment.Near;
					textBounds.Offset(-3, 0);
					e.Graphics.DrawString(_listBox.GetItemText(Items[e.Index]), e.Font, new SolidBrush(fg), textBounds);
				}
				return;
			}
            if (e.Index >= 0)
            {
                if (DrawItem != null)
                {
                    DrawItem(this, e);
                }
            }
        }

		void _listBox_SelectedIndexChanged(object sender, EventArgs e)
		{
			OnSelectedIndexChanged(e);
		}

		void _listBox_SelectedValueChanged(object sender, EventArgs e)
		{
			OnSelectedValueChanged(e);
		}

        void _listBox_MeasureItem(object sender, MeasureItemEventArgs e)
        {
            if (MeasureItem != null)
            {
                MeasureItem(this, e);
            }
        }


        void _popupControl_Closed(object sender, ToolStripDropDownClosedEventArgs e)
        {
			_isDroppedDown = false;
            if (!this.RectangleToScreen(this.ClientRectangle).Contains(MousePosition))
            {
                hovered = false;
            }
			_hoverItem = -1;
            Invalidate(true);
        }



        void _textBox_Resize(object sender, EventArgs e)
        {
            this.AdjustControls();
        }

        void _textBox_TextChanged(object sender, EventArgs e)
        {
            OnTextChanged(e);
        }

        #endregion



		public override Size MinimumSize
		{
			get
			{
				return new Size(40 + (this.ImageList != null ? this.ImageList.ImageSize.Width + 5 : 0), _textBox != null ? _textBox.Height + 8 : 21);
			}
		}

        #region PrivateMethods

        private void AdjustControls()
        {
            this.SuspendLayout();

            resize = true;
            _textBox.Top = 4;
			_textBox.Left = 5 + (this.ImageList != null ? this.ImageList.ImageSize.Width + 5 : 0);

            this.Height = _textBox.Top + _textBox.Height + _textBox.Top;

            rectBtn =
                    new System.Drawing.Rectangle(this.ClientRectangle.Width - 18,
                    this.ClientRectangle.Top, 18, _textBox.Height + 2 * _textBox.Top);


            _textBox.Width = rectBtn.Left - 1 - _textBox.Left;

            rectContent = new Rectangle(ClientRectangle.Left, ClientRectangle.Top,
                ClientRectangle.Width, _textBox.Height + 2 * _textBox.Top);

			if (_listBox != null)
			{
				_listBox.ItemHeight = _textBox.Height;
				if (this.ImageList != null)
					_listBox.ItemHeight = Math.Max(_listBox.ItemHeight, this.ImageList.ImageSize.Height);
			}

            this.ResumeLayout();

            Invalidate(true);
        }

		private int CalculateListHeight()
		{
			int borderHeight = _listBox.BorderStyle == BorderStyle.None ? 3 : SystemInformation.BorderSize.Height * 4 + 3;

			if (_listBox.Items.Count <= 0)
				return 15 + borderHeight;

			int h = 0;
			int i = 0;
			int maxItemHeight = 0;
			int highestItemHeight = 0;
			foreach (object item in _listBox.Items)
			{
				int itHeight = _listBox.GetItemHeight(i);
				if (highestItemHeight < itHeight)
				{
					highestItemHeight = itHeight;
				}
				h = h + itHeight;
				if (i <= (_maxDropDownItems - 1))
				{
					maxItemHeight = h;
				}
				i = i + 1;
			}

			if (maxItemHeight > _dropDownHeight)
				return _dropDownHeight + borderHeight;
			if (maxItemHeight > highestItemHeight)
				return maxItemHeight + borderHeight;
			return highestItemHeight + borderHeight;
		}

        private Point CalculateDropPosition()
        {
            Point point = new Point(0, this.Height);
            if ((this.PointToScreen(new Point(0, 0)).Y + this.Height + _controlHost.Height) > Screen.PrimaryScreen.WorkingArea.Height)
            {
                point.Y = -this._controlHost.Height - 7;
            }
            return point;
        }

        private Point CalculateDropPosition(int myHeight, int controlHostHeight)
        {
            Point point = new Point(0, myHeight);
            if ((this.PointToScreen(new Point(0, 0)).Y + this.Height + controlHostHeight) > Screen.PrimaryScreen.WorkingArea.Height)
            {
                point.Y = -controlHostHeight - 7;
            }
            return point;
        }

        #endregion      



        
        #region VirtualMethods

        public virtual void OnDroppedDown(object sender, EventArgs e)
        {
            if (DroppedDown != null)
            {
                DroppedDown(this, e);
            }
        }

        #endregion

        #region Render

        public static GraphicsPath CreateRoundRectangle(Rectangle rectangle, int topLeftRadius, int topRightRadius,
            int bottomRightRadius, int bottomLeftRadius)
        {
            GraphicsPath path = new GraphicsPath();
            int l = rectangle.Left;
            int t = rectangle.Top;
            int w = rectangle.Width;
            int h = rectangle.Height;

            if(topLeftRadius > 0)
            {
                path.AddArc(l, t, topLeftRadius * 2, topLeftRadius * 2, 180, 90);
            }
            path.AddLine(l + topLeftRadius, t, l + w - topRightRadius, t);
            if (topRightRadius > 0)
            {
                path.AddArc(l + w - topRightRadius * 2, t, topRightRadius * 2, topRightRadius * 2, 270, 90);
            }
            path.AddLine(l + w, t + topRightRadius, l + w, t + h - bottomRightRadius);
            if (bottomRightRadius > 0)
            {
                path.AddArc(l + w - bottomRightRadius * 2, t + h - bottomRightRadius * 2,
                    bottomRightRadius * 2, bottomRightRadius * 2, 0, 90);
            }
            path.AddLine(l + w - bottomRightRadius, t + h, l + bottomLeftRadius, t + h);
            if(bottomLeftRadius >0)
            {
                path.AddArc(l, t + h - bottomLeftRadius * 2, bottomLeftRadius * 2, bottomLeftRadius * 2, 90, 90);
            }
            path.AddLine(l, t + h - bottomLeftRadius, l, t + topLeftRadius);
            path.CloseFigure();
            return path;
        }

        #endregion
    }

	[Serializable]
    public class BNRadius
    {
        private int _topLeft = 0;

		public static readonly BNRadius Default = new BNRadius();

        public int TopLeft
        {
            get { return _topLeft; }
            set { _topLeft = value; }
        }

        private int _topRight = 0;

        public int TopRight
        {
            get { return _topRight; }
            set { _topRight = value; }
        }

        private int _bottomLeft = 0;

        public int BottomLeft
        {
            get { return _bottomLeft; }
            set { _bottomLeft = value; }
        }

        private int _bottomRight = 0;

        public int BottomRight
        {
            get { return _bottomRight; }
            set { _bottomRight = value; }
        }
    }
    
}
