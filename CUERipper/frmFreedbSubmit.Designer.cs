namespace CUERipper
{
	partial class frmFreedbSubmit
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.components = new System.ComponentModel.Container();
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(frmFreedbSubmit));
			CUEControls.RectRadius rectRadius1 = new CUEControls.RectRadius();
			this.imgComboBoxCategory = new CUEControls.ImgComboBox();
			this.frmFreedbSubmitDataBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.categoriesBindingSource = new System.Windows.Forms.BindingSource(this.components);
			this.buttonCancel = new System.Windows.Forms.Button();
			this.buttonOk = new System.Windows.Forms.Button();
			this.labelCategory = new System.Windows.Forms.Label();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.labelAT = new System.Windows.Forms.Label();
			this.labelEmail = new System.Windows.Forms.Label();
			this.textBoxDomain = new System.Windows.Forms.TextBox();
			this.textBoxUser = new System.Windows.Forms.TextBox();
			((System.ComponentModel.ISupportInitialize)(this.frmFreedbSubmitDataBindingSource)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.categoriesBindingSource)).BeginInit();
			this.groupBox1.SuspendLayout();
			this.SuspendLayout();
			// 
			// imgComboBoxCategory
			// 
			this.imgComboBoxCategory.BackColor = System.Drawing.Color.Transparent;
			this.imgComboBoxCategory.DataBindings.Add(new System.Windows.Forms.Binding("SelectedItem", this.frmFreedbSubmitDataBindingSource, "Category", true));
			this.imgComboBoxCategory.DataSource = this.categoriesBindingSource;
			this.imgComboBoxCategory.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.imgComboBoxCategory.FormattingEnabled = true;
			this.imgComboBoxCategory.ImageList = null;
			resources.ApplyResources(this.imgComboBoxCategory, "imgComboBoxCategory");
			this.imgComboBoxCategory.Name = "imgComboBoxCategory";
			rectRadius1.BottomLeft = 2;
			rectRadius1.BottomRight = 2;
			rectRadius1.TopLeft = 2;
			rectRadius1.TopRight = 6;
			this.imgComboBoxCategory.Radius = rectRadius1;
			this.imgComboBoxCategory.Text = null;
			// 
			// frmFreedbSubmitDataBindingSource
			// 
			this.frmFreedbSubmitDataBindingSource.DataSource = typeof(CUERipper.frmFreedbSubmitData);
			// 
			// categoriesBindingSource
			// 
			this.categoriesBindingSource.DataMember = "Categories";
			this.categoriesBindingSource.DataSource = this.frmFreedbSubmitDataBindingSource;
			// 
			// buttonCancel
			// 
			this.buttonCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			resources.ApplyResources(this.buttonCancel, "buttonCancel");
			this.buttonCancel.Name = "buttonCancel";
			this.buttonCancel.UseVisualStyleBackColor = true;
			// 
			// buttonOk
			// 
			this.buttonOk.DialogResult = System.Windows.Forms.DialogResult.OK;
			resources.ApplyResources(this.buttonOk, "buttonOk");
			this.buttonOk.Name = "buttonOk";
			this.buttonOk.UseVisualStyleBackColor = true;
			// 
			// labelCategory
			// 
			resources.ApplyResources(this.labelCategory, "labelCategory");
			this.labelCategory.Name = "labelCategory";
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add(this.labelAT);
			this.groupBox1.Controls.Add(this.labelEmail);
			this.groupBox1.Controls.Add(this.textBoxDomain);
			this.groupBox1.Controls.Add(this.textBoxUser);
			this.groupBox1.Controls.Add(this.labelCategory);
			this.groupBox1.Controls.Add(this.imgComboBoxCategory);
			resources.ApplyResources(this.groupBox1, "groupBox1");
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.TabStop = false;
			// 
			// labelAT
			// 
			resources.ApplyResources(this.labelAT, "labelAT");
			this.labelAT.BackColor = System.Drawing.Color.Transparent;
			this.labelAT.Name = "labelAT";
			// 
			// labelEmail
			// 
			resources.ApplyResources(this.labelEmail, "labelEmail");
			this.labelEmail.Name = "labelEmail";
			// 
			// textBoxDomain
			// 
			this.textBoxDomain.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.frmFreedbSubmitDataBindingSource, "Domain", true));
			resources.ApplyResources(this.textBoxDomain, "textBoxDomain");
			this.textBoxDomain.Name = "textBoxDomain";
			// 
			// textBoxUser
			// 
			this.textBoxUser.DataBindings.Add(new System.Windows.Forms.Binding("Text", this.frmFreedbSubmitDataBindingSource, "User", true));
			resources.ApplyResources(this.textBoxUser, "textBoxUser");
			this.textBoxUser.Name = "textBoxUser";
			// 
			// frmFreedbSubmit
			// 
			this.AcceptButton = this.buttonOk;
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.CancelButton = this.buttonCancel;
			this.Controls.Add(this.groupBox1);
			this.Controls.Add(this.buttonOk);
			this.Controls.Add(this.buttonCancel);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.MaximizeBox = false;
			this.MinimizeBox = false;
			this.Name = "frmFreedbSubmit";
			this.Load += new System.EventHandler(this.frmFreedbSubmit_Load);
			((System.ComponentModel.ISupportInitialize)(this.frmFreedbSubmitDataBindingSource)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.categoriesBindingSource)).EndInit();
			this.groupBox1.ResumeLayout(false);
			this.groupBox1.PerformLayout();
			this.ResumeLayout(false);

		}

		#endregion

		private CUEControls.ImgComboBox imgComboBoxCategory;
		private System.Windows.Forms.Button buttonCancel;
		private System.Windows.Forms.Button buttonOk;
		private System.Windows.Forms.Label labelCategory;
		private System.Windows.Forms.GroupBox groupBox1;
		private System.Windows.Forms.TextBox textBoxDomain;
		private System.Windows.Forms.TextBox textBoxUser;
		private System.Windows.Forms.Label labelEmail;
		private System.Windows.Forms.BindingSource frmFreedbSubmitDataBindingSource;
		private System.Windows.Forms.BindingSource categoriesBindingSource;
		private System.Windows.Forms.Label labelAT;
	}
}