using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Globalization;
using System.IO;
using System.Threading;
using CUETools.Processor;
using CUEControls;

namespace JDP {
	public partial class frmSettings : Form {
		bool _reducePriority;
		CUEConfig _config;
		private ShellIconMgr m_icon_mgr;

		public frmSettings() {
			InitializeComponent();
		}

		public ShellIconMgr IconMgr
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

		private void frmSettings_Load(object sender, EventArgs e) {
			chkReducePriority.Checked = _reducePriority;
			chkPreserveHTOA.Checked = _config.preserveHTOA;
			chkAutoCorrectFilenames.Checked = _config.autoCorrectFilenames;
			numericFLACCompressionLevel.Value = _config.flacCompressionLevel;
			numFixWhenConfidence.Value = _config.fixOffsetMinimumConfidence;
			numFixWhenPercent.Value = _config.fixOffsetMinimumTracksPercent;
			numEncodeWhenConfidence.Value = _config.encodeWhenConfidence;
			numEncodeWhenPercent.Value = _config.encodeWhenPercent;
			chkEncodeWhenZeroOffset.Checked = _config.encodeWhenZeroOffset;
			chkFLACVerify.Checked = _config.flacVerify;
			chkWriteArTagsOnConvert.Checked = _config.writeArTagsOnConvert;
			chkWriteARTagsOnVerify.Checked = _config.writeArTagsOnVerify;
			if (_config.wvCompressionMode == 0) rbWVFast.Checked = true;
			if (_config.wvCompressionMode == 1) rbWVNormal.Checked = true;
			if (_config.wvCompressionMode == 2) rbWVHigh.Checked = true;
			if (_config.wvCompressionMode == 3) rbWVVeryHigh.Checked = true;
			chkWVExtraMode.Checked = (_config.wvExtraMode != 0);
			if (_config.wvExtraMode != 0) numWVExtraMode.Value = _config.wvExtraMode;
			chkWVStoreMD5.Checked = _config.wvStoreMD5;
			switch (_config.apeCompressionLevel)
			{
				case 1: rbAPEfast.Checked = true; break;
				case 2: rbAPEnormal.Checked = true; break;
				case 3: rbAPEhigh.Checked = true; break;
				case 4: rbAPEextrahigh.Checked = true; break;
				case 5: rbAPEinsane.Checked = true; break;
			}
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
			numericLossyWAVQuality.Value = _config.lossyWAVQuality;
			chkHDCDLW16.Checked = _config.decodeHDCDtoLW16;
			chkHDCD24bit.Checked = _config.decodeHDCDto24bit;
			chkOverwriteTags.Checked = _config.overwriteCUEData;
			chkAllowMultipleInstances.Checked = !_config.oneInstance;
			checkBoxWriteCUETags.Checked = _config.writeBasicTagsFromCUEData;
			checkBoxCopyBasicTags.Checked = _config.copyBasicTags;
			checkBoxCopyUnknownTags.Checked = _config.copyUnknownTags;
			checkBoxCopyAlbumArt.Checked = _config.copyAlbumArt;
			checkBoxEmbedAlbumArt.Checked = _config.embedAlbumArt;
			checkBoxARVerifyUseSourceFolder.Checked = _config.arLogToSourceFolder;
			checkBoxARLogVerbose.Checked = _config.arLogVerbose;
			checkBoxFixToNearest.Checked = _config.fixOffsetToNearest;
			textBoxARLogExtension.Text = _config.arLogExtension;
			numericUpDownMaxResolution.Value = _config.maxAlbumArtSize;

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
			
			foreach (KeyValuePair<string, CUEToolsUDC> encoder in _config.encoders)
			{
				ListViewItem item = new ListViewItem(encoder.Key);
				item.Tag = encoder.Value;
				listViewEncoders.Items.Add(item);
			}
			//listViewEncoders.Items[0].Selected = true;
			foreach (KeyValuePair<string, CUEToolsUDC> decoder in _config.decoders)
				if (decoder.Value.path != null)
				{
					ListViewItem item = new ListViewItem(decoder.Key);
					item.Tag = decoder.Value;
					listViewDecoders.Items.Add(item);
				}
			//listViewDecoders.Items[0].Selected = true;
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
			listViewScriptConditions.Items[0].Tag = CUEAction.Verify;
			listViewScriptConditions.Items[1].Tag = CUEAction.VerifyAndConvert;
			listViewScriptConditions.Items[2].Tag = CUEAction.Convert;

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
			if (listViewEncoders.SelectedIndices.Count > 0)
				listViewEncoders.Items[listViewEncoders.SelectedIndices[0]].Selected = false;
			if (listViewDecoders.SelectedIndices.Count > 0)
				listViewDecoders.Items[listViewDecoders.SelectedIndices[0]].Selected = false;
			if (listViewScripts.SelectedItems.Count > 0)
				listViewScripts.SelectedItems[0].Selected = false;

			_reducePriority = chkReducePriority.Checked;
			_config.preserveHTOA = chkPreserveHTOA.Checked;
			_config.autoCorrectFilenames = chkAutoCorrectFilenames.Checked;
			_config.flacCompressionLevel = (uint)numericFLACCompressionLevel.Value;
			_config.lossyWAVQuality = (int)numericLossyWAVQuality.Value;
			_config.fixOffsetMinimumTracksPercent = (uint)numFixWhenPercent.Value;
			_config.fixOffsetMinimumConfidence = (uint)numFixWhenConfidence.Value;
			_config.encodeWhenPercent = (uint)numEncodeWhenPercent.Value;
			_config.encodeWhenConfidence = (uint)numEncodeWhenConfidence.Value;
			_config.encodeWhenZeroOffset = chkEncodeWhenZeroOffset.Checked;
			_config.flacVerify = chkFLACVerify.Checked;
			_config.writeArTagsOnConvert = chkWriteArTagsOnConvert.Checked;
			_config.writeArTagsOnVerify = chkWriteARTagsOnVerify.Checked;
			if (rbWVFast.Checked) _config.wvCompressionMode = 0;
			else if (rbWVHigh.Checked) _config.wvCompressionMode = 2;
			else if (rbWVVeryHigh.Checked) _config.wvCompressionMode = 3;
			else _config.wvCompressionMode = 1;
			if (!chkWVExtraMode.Checked) _config.wvExtraMode = 0;
			else _config.wvExtraMode = (int) numWVExtraMode.Value;
			_config.wvStoreMD5 = chkWVStoreMD5.Checked;
			_config.apeCompressionLevel = (uint) (rbAPEfast.Checked ? 1 :
				rbAPEnormal.Checked ? 2 :
				rbAPEhigh.Checked ? 3 :
				rbAPEextrahigh.Checked ? 4 :
				rbAPEinsane.Checked ? 5 : 2);
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
			_config.copyAlbumArt = checkBoxCopyAlbumArt.Checked;
			_config.embedAlbumArt = checkBoxEmbedAlbumArt.Checked;

			_config.arLogToSourceFolder = checkBoxARVerifyUseSourceFolder.Checked;
			_config.arLogVerbose = checkBoxARLogVerbose.Checked;
			_config.fixOffsetToNearest = checkBoxFixToNearest.Checked;
			_config.arLogExtension = textBoxARLogExtension.Text;
			_config.maxAlbumArtSize = (int) numericUpDownMaxResolution.Value;

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

			numWVExtraMode.Enabled = chkWVExtraMode.Checked;

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
			if (listViewEncoders.SelectedItems.Count > 0)
				listViewEncoders.SelectedItems[0].Selected = false;
			if (listViewDecoders.SelectedItems.Count > 0)
				listViewDecoders.SelectedItems[0].Selected = false;
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

			foreach (KeyValuePair<string, CUEToolsUDC> encoder in _config.encoders)
				if (encoder.Value.extension == format.extension)
					encoder.Value.extension = e.Label;

			foreach (KeyValuePair<string, CUEToolsUDC> decoder in _config.decoders)
				if (decoder.Value.extension == format.extension)
					decoder.Value.extension = e.Label;

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
						format = new CUEToolsFormat("new", CUEToolsTagger.TagLibSharp, true, true, false, false, false, null, null, null);
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
						List<string> decodersToRemove = new List<string>();
						foreach (KeyValuePair<string, CUEToolsUDC> decoder in _config.decoders)
							if (decoder.Value.extension == format.extension)
								decodersToRemove.Add(decoder.Key);
						foreach (string decoder in decodersToRemove)
						{
							_config.decoders.Remove(decoder);
							foreach (ListViewItem item in listViewDecoders.Items)
								if (item.Text == decoder)
								{
									item.Remove();
									break;
								}
						}
						List<string> encodersToRemove = new List<string>();
						foreach (KeyValuePair<string, CUEToolsUDC> encoder in _config.encoders)
							if (encoder.Value.extension == format.extension)
								encodersToRemove.Add(encoder.Key);
						foreach (string encoder in encodersToRemove)
						{
							_config.encoders.Remove(encoder);
							foreach (ListViewItem item in listViewEncoders.Items)
								if (item.Text == encoder)
								{
									item.Remove();
									break;
								}
						}
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
				foreach (KeyValuePair<string, CUEToolsUDC> encoder in _config.encoders)
					if (encoder.Value.extension == format.extension && encoder.Value.lossless)
						comboFormatLosslessEncoder.Items.Add(encoder.Key);
				comboFormatLosslessEncoder.SelectedItem = format.encoderLossless;
				comboFormatLosslessEncoder.Enabled = format.allowLossless;

				comboFormatLossyEncoder.Items.Clear();
				foreach (KeyValuePair<string, CUEToolsUDC> encoder in _config.encoders)
					if (encoder.Value.extension == format.extension && !encoder.Value.lossless)
						comboFormatLossyEncoder.Items.Add(encoder.Key);
				comboFormatLossyEncoder.SelectedItem = format.encoderLossy;
				comboFormatLossyEncoder.Enabled = format.allowLossy;

				comboFormatDecoder.Items.Clear();
				foreach (KeyValuePair<string, CUEToolsUDC> decoder in _config.decoders)
					if (decoder.Value.extension == format.extension)
						comboFormatDecoder.Items.Add(decoder.Key);
				comboFormatDecoder.SelectedItem = format.decoder;
				comboFormatDecoder.Enabled = format.allowLossless;

				comboBoxFormatTagger.SelectedItem = format.tagger;

				checkBoxFormatEmbedCUESheet.Checked = format.allowEmbed;
				checkBoxFormatAllowLossless.Checked = format.allowLossless;
				checkBoxFormatAllowLossy.Checked = format.allowLossy;
				checkBoxFormatSupportsLossyWAV.Checked = format.allowLossyWAV;

				comboBoxFormatTagger.Enabled =
					checkBoxFormatEmbedCUESheet.Enabled =
					checkBoxFormatSupportsLossyWAV.Enabled =
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

				format.encoderLossless = (string)comboFormatLosslessEncoder.SelectedItem;
				format.encoderLossy = (string)comboFormatLossyEncoder.SelectedItem;
				format.decoder = (string)comboFormatDecoder.SelectedItem;
				if (!format.builtin)
				{
					format.tagger = (CUEToolsTagger)comboBoxFormatTagger.SelectedItem;
					format.allowEmbed = checkBoxFormatEmbedCUESheet.Checked;
					format.allowLossyWAV = checkBoxFormatSupportsLossyWAV.Checked;
					format.allowLossless = checkBoxFormatAllowLossless.Checked;
					format.allowLossy = checkBoxFormatAllowLossy.Checked;
				}
			}			
		}

		private void comboBoxEncoderExtension_SelectedIndexChanged(object sender, EventArgs e)
		{
			labelEncoderExtension.ImageKey = "." + (string)comboBoxEncoderExtension.SelectedItem;
			CUEToolsFormat format;
			if (_config.formats.TryGetValue((string)comboBoxEncoderExtension.SelectedItem, out format))
			{
				checkBoxEncoderLossless.Enabled = format.allowLossless && format.allowLossy;
				if (!checkBoxEncoderLossless.Enabled)
					checkBoxEncoderLossless.Checked = format.allowLossless;
			}
		}

		private void listViewEncoders_ItemSelectionChanged(object sender, ListViewItemSelectionChangedEventArgs e)
		{
			if (e.IsSelected)
			{
				CUEToolsUDC encoder = (CUEToolsUDC)e.Item.Tag;
				if (encoder == null) return;

				comboBoxEncoderExtension.Visible = true;
				comboBoxEncoderExtension.SelectedItem = encoder.extension;
				labelEncoderExtension.Visible = true;
				if (encoder.path != null)
				{
					CUEToolsFormat format;
					comboBoxEncoderExtension.Enabled = true;
					groupBoxExternalEncoder.Visible = true;
					textBoxEncoderPath.Text = encoder.path;
					textBoxEncoderParameters.Text = encoder.parameters;
					checkBoxEncoderLossless.Checked = encoder.lossless;
					checkBoxEncoderLossless.Enabled = _config.formats.TryGetValue(encoder.extension, out format) && format.allowLossless && format.allowLossy;
				}
				else
				{
					comboBoxEncoderExtension.Enabled = false;
					switch (encoder.className)
					{
						case "FLACWriter":
							groupBoxLibFLAC.Visible = true;
							break;
						case "WavPackWriter":
							groupBoxLibWavpack.Visible = true;
							break;
						case "APEWriter":
							groupBoxLibMAC_SDK.Visible = true;
							break;
					}
				}
			}
			else
			{
				CUEToolsUDC encoder = (CUEToolsUDC)e.Item.Tag;
				if (encoder == null) return;

				if (encoder.path != null)
				{
					if (encoder.extension != (string)comboBoxEncoderExtension.SelectedItem || encoder.lossless != checkBoxEncoderLossless.Checked)
					{
						if (listViewFormats.SelectedItems.Count > 0)
							listViewFormats.SelectedItems[0].Selected = false;
						CUEToolsFormat format;
						if (_config.formats.TryGetValue(encoder.extension, out format))
						{
							if (format.encoderLossless == encoder.name)
								format.encoderLossless = null;
							if (format.encoderLossy == encoder.name)
								format.encoderLossy = null;
						}
						encoder.extension = (string)comboBoxEncoderExtension.SelectedItem;
					}
					encoder.path = textBoxEncoderPath.Text;
					encoder.parameters = textBoxEncoderParameters.Text;
					encoder.lossless = checkBoxEncoderLossless.Checked;
				}

				comboBoxEncoderExtension.Visible = false;
				labelEncoderExtension.Visible = false;
				groupBoxExternalEncoder.Visible = false;
				groupBoxLibFLAC.Visible = false;
				groupBoxLibWavpack.Visible = false;
				groupBoxLibMAC_SDK.Visible = false;
			}
		}

		private void listViewEncoders_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsUDC encoder = (CUEToolsUDC)listViewEncoders.Items[e.Item].Tag;
			if (encoder.path == null)
				e.CancelEdit = true;
		}

		private void listViewEncoders_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsUDC encoder;
			if (e.Label == null || _config.encoders.TryGetValue(e.Label, out encoder))
			{
				e.CancelEdit = true;
				return;
			}
			encoder = (CUEToolsUDC) listViewEncoders.Items[e.Item].Tag;
			if (listViewFormats.SelectedItems.Count > 0)
				listViewFormats.SelectedItems[0].Selected = false;
			if (_config.formats[encoder.extension].encoderLossless == encoder.name)
				_config.formats[encoder.extension].encoderLossless = e.Label;
			if (_config.formats[encoder.extension].encoderLossy == encoder.name)
				_config.formats[encoder.extension].encoderLossy = e.Label;
			_config.encoders.Remove(encoder.name);
			encoder.name = e.Label;
			_config.encoders.Add(encoder.name, encoder);
		}

		private void listViewEncoders_KeyDown(object sender, KeyEventArgs e)
		{
			switch (e.KeyCode)
			{
				case Keys.Insert:
					{
						CUEToolsUDC encoder;
						if (_config.encoders.TryGetValue("new", out encoder))
							return;
						encoder = new CUEToolsUDC("new", "wav", true, "", "");
						_config.encoders.Add("new", encoder);
						ListViewItem item = new ListViewItem(encoder.name);
						item.Tag = encoder;
						listViewEncoders.Items.Add(item);
						item.BeginEdit();
						break;
					}
				case Keys.Delete:
					{
						if (listViewEncoders.SelectedItems.Count <= 0)
							return;
						CUEToolsUDC encoder = (CUEToolsUDC)listViewEncoders.SelectedItems[0].Tag;
						if (encoder.path == null)
							return;
						if (_config.formats[encoder.extension].encoderLossless == encoder.name)
							_config.formats[encoder.extension].encoderLossless = null;
						if (_config.formats[encoder.extension].encoderLossy == encoder.name)
							_config.formats[encoder.extension].encoderLossy = null;
						_config.encoders.Remove(encoder.name);
						listViewEncoders.Items.Remove(listViewEncoders.SelectedItems[0]);
						break;
					}
			}
		}

		private void listViewDecoders_ItemSelectionChanged(object sender, ListViewItemSelectionChangedEventArgs e)
		{
			if (e.IsSelected)
			{
				CUEToolsUDC decoder = (CUEToolsUDC)e.Item.Tag;
				if (decoder == null) return;
				comboBoxDecoderExtension.SelectedItem = decoder.extension;
				comboBoxDecoderExtension.Visible = true;
				labelDecoderExtension.Visible = true;
				if (decoder.path != null)
				{
					comboBoxDecoderExtension.Enabled = true;
					groupBoxExternalDecoder.Visible = true;
					textBoxDecoderPath.Text = decoder.path;
					textBoxDecoderParameters.Text = decoder.parameters;
				}
				else
				{
					comboBoxDecoderExtension.Enabled = false;
				}
			}
			else
			{
				CUEToolsUDC decoder = (CUEToolsUDC)e.Item.Tag;
				if (decoder == null) return;

				if (decoder.path != null)
				{
					decoder.path = textBoxDecoderPath.Text;
					decoder.parameters = textBoxDecoderParameters.Text;
					if (decoder.extension != (string)comboBoxDecoderExtension.SelectedItem)
					{
						if (listViewFormats.SelectedItems.Count > 0)
							listViewFormats.SelectedItems[0].Selected = false;
						CUEToolsFormat format;
						if (_config.formats.TryGetValue(decoder.extension, out format) && format.decoder == decoder.name)
							format.decoder = null;
						decoder.extension = (string)comboBoxDecoderExtension.SelectedItem;
					}
				}

				comboBoxDecoderExtension.Visible = false;
				labelDecoderExtension.Visible = false;
				groupBoxExternalDecoder.Visible = false;
			}
		}

		private void listViewDecoders_AfterLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsUDC decoder;
			if (e.Label == null || _config.decoders.TryGetValue(e.Label, out decoder))
			{
				e.CancelEdit = true;
				return;
			}
			decoder = (CUEToolsUDC)listViewDecoders.Items[e.Item].Tag;
			if (listViewFormats.SelectedItems.Count > 0)
				listViewFormats.SelectedItems[0].Selected = false;
			if (_config.formats[decoder.extension].decoder == decoder.name)
				_config.formats[decoder.extension].decoder = e.Label;
			_config.decoders.Remove(decoder.name);
			decoder.name = e.Label;
			_config.decoders.Add(decoder.name, decoder);
		}

		private void listViewDecoders_BeforeLabelEdit(object sender, LabelEditEventArgs e)
		{
			CUEToolsUDC decoder = (CUEToolsUDC)listViewDecoders.Items[e.Item].Tag;
			if (decoder.path == null)
				e.CancelEdit = true;
		}

		private void listViewDecoders_KeyDown(object sender, KeyEventArgs e)
		{
			switch (e.KeyCode)
			{
				case Keys.Insert:
					{
						CUEToolsUDC decoder;
						if (_config.decoders.TryGetValue("new", out decoder))
							return;
						decoder = new CUEToolsUDC("new", "wav", true, "", "");
						_config.decoders.Add("new", decoder);
						ListViewItem item = new ListViewItem(decoder.name);
						item.Tag = decoder;
						listViewDecoders.Items.Add(item);
						item.BeginEdit();
						break;
					}
				case Keys.Delete:
					{
						if (listViewDecoders.SelectedItems.Count <= 0)
							return;
						CUEToolsUDC decoder = (CUEToolsUDC)listViewDecoders.SelectedItems[0].Tag;
						if (decoder.path == null)
							return;
						if (_config.formats[decoder.extension].decoder == decoder.name)
							_config.formats[decoder.extension].decoder = null;
						_config.decoders.Remove(decoder.name);
						listViewDecoders.Items.Remove(listViewDecoders.SelectedItems[0]);
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
	}
}