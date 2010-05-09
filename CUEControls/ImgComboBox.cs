using System.Windows.Forms;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms.Design;
using System.ComponentModel;
using System;

namespace CUEControls
{
	public class ImgComboBox : ComboBox
	{
		private Rectangle rectBtn = new Rectangle(0, 0, 1, 1);
		private Rectangle rectContent = new Rectangle(0, 0, 1, 1);
		private Rectangle rectTextBounds = new Rectangle(0, 0, 1, 1);
		private RectRadius _radius = new RectRadius(2, 6, 2, 2);

		public ImgComboBox()
		{
			SetStyle(ControlStyles.AllPaintingInWmPaint, true);
			SetStyle(ControlStyles.OptimizedDoubleBuffer, true);
			SetStyle(ControlStyles.ResizeRedraw, true);
			SetStyle(ControlStyles.SupportsTransparentBackColor, true);
			//SetStyle(ControlStyles.UserMouse, true);
			SetStyle(ControlStyles.UserPaint, true);
			//SetStyle(ControlStyles.Selectable, true);
			base.BackColor = Color.Transparent;
			base.DrawMode = DrawMode.OwnerDrawFixed;
			AdjustControls();
		}

		#region Properties

		private string _text;

        [Localizable(true),Bindable(true)]
		public override string Text
		{
			get
			{
				if (DropDownStyle != ComboBoxStyle.DropDownList || Items.Count != 0)
					return base.Text;
				return _text;
			}
			set
			{
				if (DropDownStyle != ComboBoxStyle.DropDownList || Items.Count != 0)
					base.Text = value;
				_text = value;
				Invalidate();
			}
		}

		private DrawMode _drawMode = DrawMode.Normal;

		[DefaultValue(DrawMode.Normal)]
		[Category("Behavior"), Description("Indicates whether the user code or this control will handle the drawing of elements in the list.")]
		public new DrawMode DrawMode
		{
			get { return _drawMode; }
			set
			{
				_drawMode = value;
				base.DrawMode = value == DrawMode.Normal ? DrawMode.OwnerDrawFixed : value;
			}
		}

		[Category("Appearance"), Description("Selects the radius of combobox edges.")]
		[DefaultValue("2, 6, 2, 2")]
		public RectRadius Radius
		{
			get { return _radius; }
			set { if (value != null) _radius = value; }
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

		protected virtual Color GetOuterBorderColor()
		{
			return (Enabled) ? BackColor : BackColor;
		}

		protected virtual Color GetInnerBorderColor()
		{
			return (Enabled) ? BackColor : SystemColors.Control;
		}

		private void AdjustControls()
		{
			rectTextBounds.X = ClientRectangle.Left + 5 + (ImageList != null ? ImageList.ImageSize.Width + 5 : 0);
			rectTextBounds.Y = ClientRectangle.Top + 4;
			rectTextBounds.Width = ClientRectangle.Width - rectTextBounds.X - 18;
			rectTextBounds.Height = ClientRectangle.Height - 8;

			rectBtn.X = ClientRectangle.Width - 18;
			rectBtn.Y = ClientRectangle.Top + 4;
			rectBtn.Width = 18;
			rectBtn.Height = ClientRectangle.Height - 8;

			rectContent.X = ClientRectangle.Left;
			rectContent.Y = ClientRectangle.Top;
			rectContent.Width = ClientRectangle.Width;
			rectContent.Height = ClientRectangle.Height;

			Invalidate(true);
		}

		private int GetImageKey(int index)
		{
			if (this.ImageList == null || index < 0)
				return -1;
			object key = FilterItemOnProperty(Items[index], ImageKeyMember ?? DisplayMember);
			if (key == null)
				return -1;
			if (key is int)
				return (int)key;
			if (key is string)
				return this.ImageList.Images.IndexOfKey(key as string);
			return -1;
		}

		protected override void OnResize(EventArgs e)
		{
			base.OnResize(e);
			AdjustControls();
		}

		protected override void OnGotFocus(EventArgs e)
		{
			base.OnGotFocus(e);
			Invalidate(true);
		}

		protected override void OnDrawItem(DrawItemEventArgs e)
		{
			if (_drawMode != DrawMode.Normal)
			{
				base.OnDrawItem(e);
				return;
			}
			if ((e.State & DrawItemState.ComboBoxEdit) != 0)
				return;
			//if (e.State == DrawItemState.
			//int _hoverItem = SelectedIndex;
			Color fg = e.ForeColor; // _hoverItem != -1 && _hoverItem == e.Index ? SystemColors.HighlightText : ForeColor;
			Color bg = e.BackColor; // _hoverItem != -1 && _hoverItem == e.Index ? SystemColors.Highlight : BackColor;

			if (bg == SystemColors.Window || bg.A < 255)
				e.Graphics.FillRectangle(SystemBrushes.Window, e.Bounds);
			else using (Brush b = new SolidBrush(bg))
				e.Graphics.FillRectangle(b, e.Bounds);

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
				//textBounds.Height = _textBox.Height;
				e.Graphics.DrawString(GetItemText(Items[e.Index]), this.Font, new SolidBrush(fg), textBounds, sf);
			}
			//base.OnDrawItem(e);
		}

		protected override void OnPaint(PaintEventArgs e)
		{
			bool hovered = this.RectangleToScreen(this.ClientRectangle).Contains(MousePosition);

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
			Color foreColor = Color.FromArgb(DroppedDown ? 100 : 50, ForeColor);
			Brush brInnerBrush = new LinearGradientBrush(
				new Rectangle(rectInner.X, rectInner.Y, rectInner.Width, rectInner.Height + 1),
				Color.FromArgb((hovered || DroppedDown || Focused) ? 200 : 100, ForeColor),
				Color.Transparent,
				LinearGradientMode.Vertical);
			Brush brBackground;
			if (this.DropDownStyle == ComboBoxStyle.DropDownList)
				brBackground = new LinearGradientBrush(pathInnerBorder.GetBounds(), BackColor, (hovered || Focused)? Color.FromArgb(100, SystemColors.HotTrack) : foreColor, LinearGradientMode.Vertical);
			else
				brBackground = new SolidBrush(BackColor);
			Pen penInnerBorder = new Pen(brInnerBrush, 0);
			LinearGradientBrush brButtonLeft = new LinearGradientBrush(rectBtn, BackColor, ForeColor, LinearGradientMode.Vertical);
			ColorBlend blend = new ColorBlend();
			blend.Colors = new Color[] { Color.Transparent, foreColor, Color.Transparent };
			blend.Positions = new float[] { 0.0f, 0.5f, 1.0f };
			brButtonLeft.InterpolationColors = blend;
			Pen penLeftButton = new Pen(brButtonLeft, 0);
			Brush brButton = new LinearGradientBrush(pathBtnBorder.GetBounds(), BackColor, foreColor, LinearGradientMode.Vertical);

			//draw
			e.Graphics.FillPath(brBackground, pathContentBorder);
			if (DropDownStyle != ComboBoxStyle.DropDownList)
			{
				e.Graphics.FillPath(brButton, pathBtnBorder);
			}
			Color outerBorderColor = GetOuterBorderColor();
			if (outerBorderColor.IsSystemColor)
			{
				Pen penOuterBorder = SystemPens.FromSystemColor(outerBorderColor);
				e.Graphics.DrawPath(penOuterBorder, pathOuterBorder);
			}
			else using (Pen penOuterBorder = new Pen(outerBorderColor))
					e.Graphics.DrawPath(penOuterBorder, pathOuterBorder);
			e.Graphics.DrawPath(penInnerBorder, pathInnerBorder);

			e.Graphics.DrawLine(penLeftButton, rectBtn.Left + 1, rectInner.Top + 1, rectBtn.Left + 1, rectInner.Bottom - 1);


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

			SolidBrush br = new SolidBrush(Enabled ? Color.Gray : Color.Gainsboro);
			e.Graphics.FillPath(br, path);
			e.Graphics.ResetTransform();
			br.Dispose();
			path.Dispose();

			// image
			if (ImageList != null)
			{
				int idx = GetImageKey(SelectedIndex);
				if (idx >= 0)
					this.ImageList.Draw(e.Graphics, rectTextBounds.Left - this.ImageList.ImageSize.Width - 5, rectContent.Y + 2, idx);
			}

			//text
			if (DropDownStyle == ComboBoxStyle.DropDownList)
			{
				StringFormat sf = new StringFormat(StringFormatFlags.NoWrap);
				sf.Alignment = StringAlignment.Near;

				Rectangle rectText = rectTextBounds;
				rectText.Offset(-3, 0);

				SolidBrush foreBrush = new SolidBrush(ForeColor);
				if (Enabled)
				{
					e.Graphics.DrawString(Text, this.Font, foreBrush, rectText, sf);
				}
				else
				{
					ControlPaint.DrawStringDisabled(e.Graphics, Text, Font, BackColor, rectText, sf);
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

			penInnerBorder.Dispose();
			penLeftButton.Dispose();

			brBackground.Dispose();
			brInnerBrush.Dispose();
			brButtonLeft.Dispose();
			brButton.Dispose();
		}

		public static GraphicsPath CreateRoundRectangle(Rectangle rectangle, int topLeftRadius, int topRightRadius,
			int bottomRightRadius, int bottomLeftRadius)
		{
			GraphicsPath path = new GraphicsPath();
			int l = rectangle.Left;
			int t = rectangle.Top;
			int w = rectangle.Width;
			int h = rectangle.Height;

			if (topLeftRadius > 0)
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
			if (bottomLeftRadius > 0)
			{
				path.AddArc(l, t + h - bottomLeftRadius * 2, bottomLeftRadius * 2, bottomLeftRadius * 2, 90, 90);
			}
			path.AddLine(l, t + h - bottomLeftRadius, l, t + topLeftRadius);
			path.CloseFigure();
			return path;
		}
	}

	[TypeConverter(typeof(RectRadiusConverter))]
	public class RectRadius : ICloneable
	{
		public static readonly RectRadius Default = new RectRadius();
		public RectRadius()
			: this(0,0,0,0)
		{
		}
		public RectRadius(int topLeft, int topRight, int bottomLeft, int bottomRight)
		{
			TopLeft = topLeft;
			TopRight = topRight;
			BottomLeft = bottomLeft;
			BottomRight = bottomRight;
		}
		public override bool Equals(object other)
		{
			RectRadius rrOther = other as RectRadius;
			return rrOther.TopLeft == TopLeft && rrOther.TopRight == TopRight && rrOther.BottomLeft == BottomLeft && rrOther.BottomRight == BottomRight;
		}
		public override int GetHashCode()
		{
			return base.GetHashCode();
		}
		object ICloneable.Clone()
		{
			return new RectRadius(TopLeft, TopRight, BottomLeft, BottomRight);
		}
		[RefreshProperties(RefreshProperties.All)]
		public int TopLeft { get; set; }
		[RefreshProperties(RefreshProperties.All)]
		public int TopRight {get; set; }
		[RefreshProperties(RefreshProperties.All)]
		public int BottomLeft { get; set; }
		[RefreshProperties(RefreshProperties.All)]
		public int BottomRight { get; set; }
	}

	// Summary:
	//     Converts rectangles from one data type to another. Access this class through
	//     the System.ComponentModel.TypeDescriptor.
	public class RectRadiusConverter : TypeConverter
	{
		public override PropertyDescriptorCollection GetProperties(ITypeDescriptorContext context, object value, Attribute[] attributes)
		{
			PropertyDescriptorCollection props = TypeDescriptor.GetProperties(typeof(RectRadius), attributes);
			return props.Sort(new string[] { "TopLeft", "TopRight", "BottomLeft", "BottomRight" });
		}

		public override bool GetPropertiesSupported(ITypeDescriptorContext context)
		{
			return true;
		}
	}

	public class ImgComboBoxItem<T>
	{
		string text;
		string imageKey;
		T value;

		public string ImageKey
		{
			get
			{
				return imageKey;
			}
		}

		public override string ToString()
		{
			return text ?? value.ToString();
		}

		public T Value
		{
			get
			{
				return value;
			}
		}

		public ImgComboBoxItem(string text, string imageKey, T value)
		{
			this.text = text;
			this.imageKey = imageKey;
			this.value = value;
		}
	}
}
