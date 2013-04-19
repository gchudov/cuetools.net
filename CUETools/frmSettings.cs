using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Windows.Forms;
using CUEControls;
using CUETools.Processor;

namespace JDP
{
	public partial class frmSettings : Form {
		bool _reducePriority;
		CUEConfig _config;
		private IIconManager m_icon_mgr;
        private CUEToolsUDC m_encoder;

		public frmSettings() 
        {
			InitializeComponent();
		}

        public frmSettings(CUEToolsUDC encoder)
        {
            InitializeComponent();
            m_encoder = encoder;
        }

		public IIconManager IconMgr
		{
			get
			{
				return m_icon_mgr;
			}
			set
			{
				m_icon_mgr = value;
			}
		}	

		private void frmSettings_Load(object sender, EventArgs e) 
		{
			cUEConfigBindingSource.DataSource = _config;
			encodersBindingSource.DataMember = "Encoders"; // for MONO bug (setting BindingSource.DataSource clears DataMember:(
			propertyGrid1.SelectedObject = _config.advanced;
			chkReducePriority.Checked = _reducePriority;
			checkBoxCheckForUpdates.Checked = _config.checkForUpdates;
			chkAutoCorrectFilenames.Checked = _config.autoCorrectFilenames;
			numFixWhenConfidence.Value = _config.fixOffsetMinimumConfidence;
			numFixWhenPercent.Value = _config.fixOffsetMinimumTracksPercent;
			numEncodeWhenConfidence.Value = _config.encodeWhenConfidence;
			numEncodeWhenPercent.Value = _config.encodeWhenPercent;
			chkEncodeWhenZeroOffset.Checked = _config.encodeWhenZeroOffset;
			chkWriteArTagsOnConvert.Checked = _config.writeArTagsOnEncode;
			chkWriteARTagsOnVerify.Checked = _config.writeArTagsOnVerify;
			chkKeepOriginalFilenames.Checked = _config.keepOriginalFilenames;
			txtSingleFilenameFormat.Text = _config.singleFilenameFormat;
			txtTrackFilenameFormat.Text = _config.trackFilenameFormat;
			chkRemoveSpecial.Checked = _config.removeSpecial;
			txtSpecialExceptions.Text = _config.specialExceptions;
			chkReplaceSpaces.Checked = _config.replaceSpaces;
			chkWriteArLogOnConvert.Checked = _config.writeArLogOnConvert;
			chkWriteARLogOnVerify.Checked = _config.writeArLogOnVerify;
			chkEmbedLog.Checked = _config.embedLog;
			chkExtractLog.Checked = _config.extractLog;
			chkFillUpCUE.Checked = _config.fillUpCUE;
			chkFilenamesANSISafe.Checked = _config.filenamesANSISafe;
			chkHDCDDetect.Checked = _config.detectHDCD;
			chkHDCDDecode.Checked = _config.decodeHDCD;
			chkHDCDStopLooking.Checked = _config.wait750FramesForHDCD;
			chkCreateM3U.Checked = _config.createM3U;
			chkCreateCUEFileWhenEmbedded.Checked = _config.createCUEFileWhenEmbedded;
			chkTruncateExtra4206Samples.Checked = _config.truncate4608ExtraSamples;
			chkHDCDLW16.Checked = _config.decodeHDCDtoLW16;
			chkHDCD24bit.Checked = _config.decodeHDCDto24bit;
			chkOverwriteTags.Checked = _config.overwriteCUEData;
			chkAllowMultipleInstances.Checked = !_config.oneInstance;
			checkBoxWriteCUETags.Checked = _config.writeBasicTagsFromCUEData;
			checkBoxCopyBasicTags.Checked = _config.copyBasicTags;
			checkBoxCopyUnknownTags.Checked = _config.copyUnknownTags;
			//checkBoxCopyAlbumArt.Checked = _config.copyAlbumArt;
			checkBoxExtractAlbumArt.Checked = _config.extractAlbumArt;
			checkBoxEmbedAlbumArt.Checked = _config.embedAlbumArt;
			checkBoxARVerifyUseSourceFolder.Checked = _config.arLogToSourceFolder;
			checkBoxARLogVerbose.Checked = _config.arLogVerbose;
			checkBoxFixToNearest.Checked = _config.fixOffsetToNearest;
			//textBoxARLogExtension.Text = _config.arLogFilenameFormat;
			numericUpDownMaxResolution.Value = _config.maxAlbumArtSize;
			checkBoxSeparateDecodingThread.Checked = _config.separateDecodingThread;

			switch (_config.gapsHandling)
			{
				case CUEStyle.GapsAppended:
					if (_config.preserveHTOA)
						rbGapsPlusHTOA.Checked = true;
					else
						rbGapsAppended.Checked = true;
					break;
				case CUEStyle.GapsPrepended:
					rbGapsPrepended.Checked = true;
					break;
				case CUEStyle.GapsLeftOut:
					rbGapsLeftOut.Checked = true;
					break;
			}

			string[] cultures = { "en-US", "de-DE", "ru-RU" };
			foreach (string culture in cultures)
			{
				try
				{
					CultureInfo info = CultureInfo.GetCultureInfo(culture);
					comboLanguage.Items.Add(info);
					if (culture == _config.language)
						comboLanguage.SelectedItem = info;
				}
				catch
				{
				}
			}
			if (comboLanguage.SelectedItem == null)
				comboLanguage.SelectedItem = comboLanguage.Items[0];
			
			listViewFormats.SmallImageList = m_icon_mgr.ImageList;
			labelEncoderExtension.ImageList = m_icon_mgr.ImageList;
			labelDecoderExtension.ImageList = m_icon_mgr.ImageList;
			foreach (KeyValuePair<string, CUEToolsFormat> format in _config.formats)
			{
				ListViewItem item = new ListViewItem(format.Key, "." + format.Key);
				item.Tag = format.Value;
				listViewFormats.Items.Add(item);
				comboBoxEncoderExtension.Items.Add(format.Key);
				comboBoxDecoderExtension.Items.Add(format.Key);
			}
			//listViewFormats.Items[0].Selected = true;
			comboBoxFormatTagger.Items.Add(CUEToolsTagger.TagLibSharp);
			comboBoxFormatTagger.Items.Add(CUEToolsTagger.APEv2);
			comboBoxFormatTagger.Items.Add(CUEToolsTagger.ID3v2);
			foreach (KeyValuePair<string, CUEToolsScript> script in _config.scripts)
			{
				ListViewItem item = new ListViewItem(script.Key);
				item.Tag = script.Value;
				listViewScripts.Items.Add(item);
			}
			ComponentResourceManager resources = new ComponentResourceManager(typeof(frmCUETools));
			listViewScriptConditions.Items.Add(resources.GetString("rbActionVerify.Text").Replace("&", ""));
			listViewScriptConditions.Items.Add(resources.GetString("rbActionEncode.Text").Replace("&", ""));
			listViewScriptConditions.Items[0].Tag = CUEAction.Verify;
			listViewScriptConditions.Items[1].Tag = CUEAction.Encode;

            if (m_encoder != null)
            {
                tabControl1.SelectedTab = tabPageEncoders;
                tabControl1.Selecting += new TabControlCancelEventHandler((s, e1) => e1.Cancel = true);
                encodersBindingSource.Position = _config.Encoders.IndexOf(m_encoder);
                listBoxEncoders.Enabled = false;
            }

            EnableDisable();
		}

		//private void DictionaryToListView(IDictionary<> dict, ListView view)
		//{
		//    foreach (KeyValuePair<string, object> format in dict)
		//    {
		//        ListViewItem item = new ListViewItem(format.Key, format.Key);
		//        item.Tag = format.Value;
		//        listViewFormats.Items.Add(item);
		//    }
		//}

		private void frmSettings_FormClosing(object sender, FormClosingEventArgs e) {
		}

		public bool ReducePriority
		{
			get { return _reducePriority; }
			set { _reducePriority = value; }
		}

		public CUEConfig Config {
			get { return _config; }
			set { _config = value; }
		}

		private void chkWVExtraMode_CheckedChanged(object sender, EventArgs e) {
			EnableDisable();
		}

		private void btnOK_Click(object sender, EventArgs e)
		{
			if (listViewFormats.SelectedIndices.Count > 0)
				listViewFormats.Items[listViewFormats.SelectedIndices[0]].Selected = false;
			if (listViewScripts.SelectedItems.Count > 0)
				listViewScripts.SelectedItems[0].Selected = false;

			_reducePriority = chkReducePriority.Checked;
			_config.checkForUpdates = checkBoxCheckForUpdates.Checked;
			_config.preserveHTOA = rbGapsPlusHTOA.Checked;
			_config.gapsHandling = rbGapsPrepended.Checked ? CUEStyle.GapsPrepended :
				rbGapsLeftOut.Checked ? CUEStyle.GapsLeftOut :
				CUEStyle.GapsAppended;
			_config.autoCorrectFilenames = chkAutoCorrectFilenames.Checked;
			_config.fixOffsetMinimumTracksPercent = (uint)numFixWhenPercent.Value;
			_config.fixOffsetMinimumConfidence = (uint)numFixWhenConfidence.Value;
			_config.encodeWhenPercent = (uint)numEncodeWhenPercent.Value;
			_config.encodeWhenConfidence = (uint)numEncodeWhenConfidence.Value;
			_config.encodeWhenZeroOffset = chkEncodeWhenZeroOffset.Checked;
			_config.writeArTagsOnEncode = chkWriteArTagsOnConvert.Checked;
			_config.writeArTagsOnVerify = chkWriteARTagsOnVerify.Checked;
			_config.keepOriginalFilenames = chkKeepOriginalFilenames.Checked;
			_config.singleFilenameFormat = txtSingleFilenameFormat.Text;
			_config.trackFilenameFormat = txtTrackFilenameFormat.Text;
			_config.removeSpecial = chkRemoveSpecial.Checked;
			_config.specialExceptions = txtSpecialExceptions.Text;
			_config.replaceSpaces = chkReplaceSpaces.Checked;
			_config.writeArLogOnConvert = chkWriteArLogOnConvert.Checked;
			_config.writeArLogOnVerify = chkWriteARLogOnVerify.Checked;
			_config.embedLog = chkEmbedLog.Checked;
			_config.extractLog = chkExtractLog.Checked;
			_config.fillUpCUE = chkFillUpCUE.Checked;
			_config.filenamesANSISafe = chkFilenamesANSISafe.Checked;
			_config.detectHDCD = chkHDCDDetect.Checked;
			_config.wait750FramesForHDCD = chkHDCDStopLooking.Checked;
			_config.decodeHDCD = chkHDCDDecode.Checked;
			_config.createM3U = chkCreateM3U.Checked;
			_config.createCUEFileWhenEmbedded = chkCreateCUEFileWhenEmbedded.Checked;
			_config.truncate4608ExtraSamples = chkTruncateExtra4206Samples.Checked;
			_config.decodeHDCDtoLW16 = chkHDCDLW16.Checked;
			_config.decodeHDCDto24bit = chkHDCD24bit.Checked;
			_config.overwriteCUEData = chkOverwriteTags.Checked;
			_config.oneInstance = !chkAllowMultipleInstances.Checked;
			_config.writeBasicTagsFromCUEData = checkBoxWriteCUETags.Checked ;
			_config.copyBasicTags = checkBoxCopyBasicTags.Checked;
			_config.copyUnknownTags = checkBoxCopyUnknownTags.Checked;
			//_config.copyAlbumArt = checkBoxCopyAlbumArt.Checked;
			_config.extractAlbumArt = checkBoxExtractAlbumArt.Checked;
			_config.embedAlbumArt = checkBoxEmbedAlbumArt.Checked;

			_config.arLogToSourceFolder = checkBoxARVerifyUseSourceFolder.Checked;
			_config.arLogVerbose = checkBoxARLogVerbose.Checked;
			_config.fixOffsetToNearest = checkBoxFixToNearest.Checked;
			//_config.arLogFilenameFormat = textBoxARLogExtension.Text;
			_config.maxAlbumArtSize = (int) numericUpDownMaxResolution.Value;
			_config.separateDecodingThread = checkBoxSeparateDecodingThread.Checked;

			_config.language = ((CultureInfo)comboLanguage.SelectedItem).Name;
		}

		private void EnableDisable()
		{
			grpHDCD.Enabled = chkHDCDDetect.Checked;
			chkHDCDLW16.Enabled = chkHDCDDetect.Checked && chkHDCDDecode.Checked;
			chkHDCD24bit.Enabled = chkHDCDDetect.Checked && chkHDCDDecode.Checked;

			chkRemoveSpecial.Enabled = chkFilenamesANSISafe.Checked;
			txtSpecialExceptions.Enabled = chkRemoveSpecial.Checked && chkFilenamesANSISafe.Checked;

			txtSpecialExceptions.Enabled = chkRemoveSpecial.Checked;

			chkOverwriteTags.Enabled = chkFillUpCUE.Checked;
		}

		private void chkArFixOffset_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkArNoUnverifiedAudio_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkHDCDDetect_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkFilenamesANSISafe_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkRemoveSpecial_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkHDCDDecode_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkFillUpCUE_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void tabControl1_Deselecting(object sender, TabControlCancelEventArgs e)
		{
			if (listViewFormats.SelectedItems.Count > 0)
				listViewFormats.SelectedItems[0].Selected = false;
			if (listViewScripts.SelectedItems.Count > 0)
				listViewScripts.SelectedItems[0].Selected = false;
		}

		private void listViewFormats_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsFormat format = (CUEToolsFormat)listViewFormats.Items[e.Item].Tag;
			if (format.builtin)
				e.CancelEdit = true;
		}

		private void listViewFormats_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsFormat format;
			if (e.Label == null || _config.formats.TryGetValue(e.Label, out format))
			{
				e.CancelEdit = true;
				return;
			}
			format = (CUEToolsFormat)listViewFormats.Items[e.Item].Tag;
			if (format.builtin)
			{
				e.CancelEdit = true;
				return;
			}

			foreach (var encoder in _config.encoders)
				if (encoder.extension == format.extension)
					encoder.extension = e.Label;

			foreach (var decoder in _config.decoders)
				if (decoder.extension == format.extension)
					decoder.extension = e.Label;

			comboBoxEncoderExtension.Items.Remove(format.extension);
			comboBoxEncoderExtension.Items.Add(e.Label);
			comboBoxDecoderExtension.Items.Remove(format.extension);
			comboBoxDecoderExtension.Items.Add(e.Label);

			_config.formats.Remove(format.extension);
			format.extension = e.Label;
			_config.formats.Add(format.extension, format);
		}

		private void listViewFormats_KeyDown(object sender, KeyEventArgs e)
		{
			switch (e.KeyCode)
			{
				case Keys.Insert:
					{
						CUEToolsFormat format;
						if (_config.formats.TryGetValue("new", out format))
							return;
						format = new CUEToolsFormat("new", CUEToolsTagger.TagLibSharp, true, true, false, false, null, null, null);
						_config.formats.Add("new", format);
						ListViewItem item = new ListViewItem(format.extension, "." + format.extension);
						item.Tag = format;
						listViewFormats.Items.Add(item);
						comboBoxEncoderExtension.Items.Add(format.extension);
						comboBoxDecoderExtension.Items.Add(format.extension);
						item.BeginEdit();
						break;
					}
				case Keys.Delete:
					{
						if (listViewFormats.SelectedItems.Count <= 0)
							return;
						CUEToolsFormat format = (CUEToolsFormat)listViewFormats.SelectedItems[0].Tag;
						if (format.builtin)
						    return;
                        var decodersToRemove = new List<CUEToolsUDC>();
						foreach (var decoder in _config.decoders)
							if (decoder.extension == format.extension)
								decodersToRemove.Add(decoder);
						foreach (var decoder in decodersToRemove)
							_config.decoders.Remove(decoder);
                        var encodersToRemove = new List<CUEToolsUDC>();
						foreach (var encoder in _config.encoders)
							if (encoder.extension == format.extension)
								encodersToRemove.Add(encoder);
						foreach (var encoder in encodersToRemove)
							_config.encoders.Remove(encoder);
						comboBoxEncoderExtension.Items.Remove(format.extension);
						comboBoxDecoderExtension.Items.Remove(format.extension);
						_config.formats.Remove(format.extension);
						listViewFormats.SelectedItems[0].Remove();
						break;
					}
			}
		}

		private void listViewFormats_ItemSelectionChanged(object sender, ListViewItemSelectionChangedEventArgs e)
		{
			if (e.IsSelected)
			{
				CUEToolsFormat format = (CUEToolsFormat)e.Item.Tag;
				if (format == null)
					return;

				comboFormatLosslessEncoder.Items.Clear();
				foreach (CUEToolsUDC encoder in _config.encoders)
					if (encoder.extension == format.extension && encoder.lossless)
						comboFormatLosslessEncoder.Items.Add(encoder);
				comboFormatLosslessEncoder.SelectedItem = format.encoderLossless;
				comboFormatLosslessEncoder.Enabled = format.allowLossless;

				comboFormatLossyEncoder.Items.Clear();
				foreach (CUEToolsUDC encoder in _config.encoders)
					if (encoder.extension == format.extension && !encoder.lossless)
						comboFormatLossyEncoder.Items.Add(encoder);
				comboFormatLossyEncoder.SelectedItem = format.encoderLossy;
				comboFormatLossyEncoder.Enabled = format.allowLossy;

				comboFormatDecoder.Items.Clear();
				foreach (var decoder in _config.decoders)
					if (decoder.extension == format.extension)
						comboFormatDecoder.Items.Add(decoder);
				comboFormatDecoder.SelectedItem = format.decoder;
				comboFormatDecoder.Enabled = format.allowLossless;

				comboBoxFormatTagger.SelectedItem = format.tagger;

				checkBoxFormatEmbedCUESheet.Checked = format.allowEmbed;
				checkBoxFormatAllowLossless.Checked = format.allowLossless;
				checkBoxFormatAllowLossy.Checked = format.allowLossy;

				comboBoxFormatTagger.Enabled =
					checkBoxFormatEmbedCUESheet.Enabled =
					checkBoxFormatAllowLossless.Enabled =
					checkBoxFormatAllowLossy.Enabled =
					!format.builtin;

				groupBoxFormat.Visible = true;
			}
			else
			{
				groupBoxFormat.Visible = false;

				CUEToolsFormat format = (CUEToolsFormat)e.Item.Tag;
				if (format == null)
					return;

				format.encoderLossless = (CUEToolsUDC)comboFormatLosslessEncoder.SelectedItem;
				format.encoderLossy = (CUEToolsUDC)comboFormatLossyEncoder.SelectedItem;
                format.decoder = (CUEToolsUDC)comboFormatDecoder.SelectedItem;
				if (!format.builtin)
				{
					format.tagger = (CUEToolsTagger)comboBoxFormatTagger.SelectedItem;
					format.allowEmbed = checkBoxFormatEmbedCUESheet.Checked;
					format.allowLossless = checkBoxFormatAllowLossless.Checked;
					format.allowLossy = checkBoxFormatAllowLossy.Checked;
				}
			}			
		}

		private void comboBoxEncoderExtension_SelectedIndexChanged(object sender, EventArgs e)
		{
			// Setting DataSourceUpdateMode to OnPropertyChanged doesn't seem 
			// to do the trick, so updating manually
			comboBoxEncoderExtension.DataBindings["SelectedItem"].WriteValue();
		}

		private void encodersBindingSource_CurrentItemChanged(object sender, EventArgs e)
		{
			CUEToolsUDC encoder = encodersBindingSource.Current as CUEToolsUDC;
			if (encoder == null)
			{
				labelEncoderExtension.Visible =
				comboBoxEncoderExtension.Visible =
				comboBoxEncoderExtension.Enabled =
				groupBoxExternalEncoder.Visible =
				checkBoxEncoderLossless.Enabled =
				propertyGridEncoderSettings.Visible = false;
				propertyGridEncoderSettings.SelectedObject = null;
			}
			else
			{
				CUEToolsFormat format = _config.formats[encoder.extension]; // _config.formats.TryGetValue(encoder.extension, out format)
				labelEncoderExtension.Visible = true;
				comboBoxEncoderExtension.Visible = true;
				comboBoxEncoderExtension.Enabled = encoder.CanBeDeleted;
                groupBoxExternalEncoder.Visible = encoder.CanBeDeleted;
				checkBoxEncoderLossless.Enabled = format != null && format.allowLossless && format.allowLossy;
                propertyGridEncoderSettings.Visible = !encoder.CanBeDeleted && encoder.settings.HasBrowsableAttributes();
                propertyGridEncoderSettings.SelectedObject = encoder.CanBeDeleted ? null : encoder.settings;
                if (!checkBoxEncoderLossless.Enabled && format != null && encoder.Lossless != format.allowLossless)
                    encoder.Lossless = format.allowLossless;
                foreach (KeyValuePair<string, CUEToolsFormat> fmtEntry in _config.formats)
				{
					CUEToolsFormat fmt = fmtEntry.Value;
					if (fmt.encoderLossless == encoder && (fmt.extension != encoder.extension || !encoder.Lossless))
						fmt.encoderLossless = null;
					if (fmt.encoderLossy == encoder && (fmt.extension != encoder.extension || encoder.Lossless))
						fmt.encoderLossy = null;
				}
			}
		}

		private void listBoxEncoders_KeyDown(object sender, KeyEventArgs e)
		{
			switch (e.KeyCode)
			{
				case Keys.Insert:
					buttonEncoderAdd_Click(sender, e);
					break;
				case Keys.Delete:
					buttonEncoderDelete_Click(sender, e);
					break;
			}
		}


		private void bindingSourceDecoders_CurrentItemChanged(object sender, EventArgs e)
		{
			var decoder = bindingSourceDecoders.Current as CUEToolsUDC;
            if (decoder == null)
            {
                labelDecoderExtension.Visible =
                comboBoxDecoderExtension.Visible = false;
            }
            else
            {
                //CUEToolsFormat format = _config.formats[decoder.extension]; // _config.formats.TryGetValue(encoder.extension, out format)
                labelDecoderExtension.Visible =
                comboBoxDecoderExtension.Visible = true;
                groupBoxExternalDecoder.Visible = decoder.CanBeDeleted;
                foreach (KeyValuePair<string, CUEToolsFormat> fmtEntry in _config.formats)
                {
                    CUEToolsFormat fmt = fmtEntry.Value;
                    if (fmt.decoder == decoder && fmt.extension != decoder.extension)
                        fmt.decoder = null;
                }
            }
        }

        private void listViewDecoders_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.Insert:
                    {
                        buttonDecoderAdd_Click(sender, e);
                        break;
                    }
                case Keys.Delete:
                    {
                        buttonDecoderDelete_Click(sender, e);
                        break;
                    }
            }
        }

		private void comboBoxDecoderExtension_SelectedIndexChanged(object sender, EventArgs e)
		{
			labelDecoderExtension.ImageKey = "." + (string)comboBoxDecoderExtension.SelectedItem;
		}

		private void listViewScripts_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsScript script = (CUEToolsScript)listViewScripts.Items[e.Item].Tag;
			if (script.builtin)
				e.CancelEdit = true;
		}

		private void listViewScripts_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsScript script;
			if (e.Label == null || _config.scripts.TryGetValue(e.Label, out script))
			{
				e.CancelEdit = true;
				return;
			}
			script = (CUEToolsScript)listViewScripts.Items[e.Item].Tag;
			if (script.builtin)
			{
				e.CancelEdit = true;
				return;
			}
			_config.scripts.Remove(script.name);
			script.name = e.Label;
			_config.scripts.Add(script.name, script);
		}

		private void listViewScripts_KeyDown(object sender, KeyEventArgs e)
		{
			switch (e.KeyCode)
			{
				case Keys.Insert:
					{
						CUEToolsScript script;
						if (_config.scripts.TryGetValue("new", out script))
							return;
						script = new CUEToolsScript("new", false, new CUEAction[] {}, "");
						_config.scripts.Add("new", script);
						ListViewItem item = new ListViewItem(script.name);
						item.Tag = script;
						listViewScripts.Items.Add(item);
						item.BeginEdit();
						break;
					}
				case Keys.Delete:
					{
						if (listViewScripts.SelectedItems.Count <= 0)
							return;
						CUEToolsScript script = (CUEToolsScript)listViewScripts.SelectedItems[0].Tag;
						if (script.builtin)
							return;
						_config.scripts.Remove(script.name);
						listViewScripts.Items.Remove(listViewScripts.SelectedItems[0]);
						break;
					}
			}
		}

		private void listViewScripts_ItemSelectionChanged(object sender, ListViewItemSelectionChangedEventArgs e)
		{
			if (e.IsSelected)
			{
				CUEToolsScript script = (CUEToolsScript)e.Item.Tag;
				if (script == null) return;
				foreach (ListViewItem item in listViewScriptConditions.Items)
					item.Checked = script.conditions.Contains((CUEAction)item.Tag);
				groupBoxScriptConditions.Visible = true;
				richTextBoxScript.Text = script.code;
				richTextBoxScript.Visible = true;
				buttonScriptCompile.Visible = true;

				groupBoxScriptConditions.Enabled =
					buttonScriptCompile.Enabled =
					!script.builtin;
				richTextBoxScript.ReadOnly = script.builtin;
			}
			else
			{
				CUEToolsScript script = (CUEToolsScript)e.Item.Tag;
				if (script == null) return;
				if (!script.builtin)
				{
					script.conditions.Clear();
					foreach (ListViewItem item in listViewScriptConditions.Items)
						if (item.Checked)
							script.conditions.Add((CUEAction)item.Tag);
					script.code = richTextBoxScript.Text;
				}
				groupBoxScriptConditions.Visible = false;
				richTextBoxScript.Visible = false;
				buttonScriptCompile.Visible = false;
			}
		}

		private static int WordLength(string text, int pos)
		{
			if (pos >= text.Length)
				return 1;
			if ((text[pos] >= 'a' && text[pos] <= 'z') ||
				(text[pos] >= 'A' && text[pos] <= 'Z') ||
				(text[pos] == '_'))
			{
				for (int len = 1; len < text.Length - pos; len++)
				{
					bool ok = (text[pos + len] >= 'a' && text[pos + len] <= 'z') ||
						(text[pos + len] >= 'A' && text[pos + len] <= 'Z') ||
						(text[pos + len] == '_');
					if (!ok)
						return len;
				}
				return text.Length - pos;
			}
			return 1;
		}

		private void buttonScriptCompile_Click(object sender, EventArgs e)
		{
			richTextBoxScript.SelectAll();
			richTextBoxScript.SelectionColor = richTextBoxScript.ForeColor;
			richTextBoxScript.DeselectAll();
			try
			{
				CUESheet.TryCompileScript(richTextBoxScript.Text);
			}
			catch (Exception ex)
			{
				using (StringWriter sw = new StringWriter())
				{
					using (StringReader sr = new StringReader(ex.Message))
					{
						string lineStr;
						while ((lineStr = sr.ReadLine()) != null)
						{
							string[] s = { ".tmp(" };
							string[] n = lineStr.Split(s, 2, StringSplitOptions.None);
							if (n.Length == 2)
							{
								string[] n2 = n[1].Split(")".ToCharArray(), 2);
								if (n2.Length == 2)
								{
									string[] n3 = n2[0].Split(",".ToCharArray(), 2);
									int row, col;
									if (n3.Length == 2 && int.TryParse(n3[0], out row) && int.TryParse(n3[1], out col) && row > 1)
									{
										int pos = richTextBoxScript.GetFirstCharIndexFromLine(row - 2);
										if (pos >= 0)
										{
											pos += col - 1;
											richTextBoxScript.Select(pos, WordLength(richTextBoxScript.Text, pos));
											richTextBoxScript.SelectionColor = Color.Red;
											richTextBoxScript.DeselectAll();
										}
									}
								}
								sw.WriteLine("({0}", n[1]);
							}
							else
								sw.WriteLine("{0}", lineStr);
						}
					}
					MessageBox.Show(this, sw.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				}
				return;
			}
			MessageBox.Show(this, "Script compiled successfully.", "Ok", MessageBoxButtons.OK, MessageBoxIcon.Information);
		}

		private void buttonEncoderAdd_Click(object sender, EventArgs e)
		{
			encodersBindingSource.AddNew();
		}

		private void buttonEncoderDelete_Click(object sender, EventArgs e)
		{
			CUEToolsUDC encoder = encodersBindingSource.Current as CUEToolsUDC;
			if (encoder == null || !encoder.CanBeDeleted)
				return;
			if (_config.formats[encoder.extension].encoderLossless == encoder)
				_config.formats[encoder.extension].encoderLossless = null;
			if (_config.formats[encoder.extension].encoderLossy == encoder)
				_config.formats[encoder.extension].encoderLossy = null;
			encodersBindingSource.RemoveCurrent();
        }

        private void buttonDecoderAdd_Click(object sender, EventArgs e)
        {
            bindingSourceDecoders.AddNew();
        }

        private void buttonDecoderDelete_Click(object sender, EventArgs e)
        {
            var decoder = bindingSourceDecoders.Current as CUEToolsUDC;
            if (decoder == null || !decoder.CanBeDeleted)
                return;
            if (_config.formats[decoder.extension].decoder == decoder)
                _config.formats[decoder.extension].decoder = null;
            bindingSourceDecoders.RemoveCurrent();
        }
	}
}