using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
    public class CUEToolsUDCEncoderList : BindingList<AudioEncoderSettingsViewModel>
    {
        public CUEToolsUDCEncoderList()
            : base()
        {
            AddingNew += OnAddingNew;
        }

        private void OnAddingNew(object sender, AddingNewEventArgs e)
        {
            e.NewObject = new AudioEncoderSettingsViewModel("new", "wav", true, "", "", "", "");
        }

        public bool TryGetValue(string extension, bool lossless, string name, out AudioEncoderSettingsViewModel result)
        {
            //result = this.Where(udc => udc.settings.Extension == extension && udc.settings.Lossless == lossless && udc.settings.Name == name).First();
            foreach (AudioEncoderSettingsViewModel udc in this)
            {
                if (udc.settings.Extension == extension && udc.settings.Lossless == lossless && udc.settings.Name == name)
                {
                    result = udc;
                    return true;
                }
            }
            result = null;
            return false;
        }

        public AudioEncoderSettingsViewModel GetDefault(string extension, bool lossless)
        {
            AudioEncoderSettingsViewModel result = null;
            foreach (AudioEncoderSettingsViewModel udc in this)
            {
                if (udc.settings.Extension == extension && udc.settings.Lossless == lossless && (result == null || result.settings.Priority < udc.settings.Priority))
                {
                    result = udc;
                }
            }
            return result;
        }
    }

    public class CUEToolsUDCDecoderList : BindingList<AudioDecoderSettingsViewModel>
    {
        public CUEToolsUDCDecoderList()
            : base()
        {
            AddingNew += OnAddingNew;
        }

        private void OnAddingNew(object sender, AddingNewEventArgs e)
        {
            e.NewObject = new AudioDecoderSettingsViewModel(new CommandLineDecoderSettings("new", "wav", "", ""));
        }

        public bool TryGetValue(string extension, bool lossless, string name, out AudioDecoderSettingsViewModel result)
        {
            foreach (AudioDecoderSettingsViewModel udc in this)
            {
                if (udc.decoderSettings.Extension == extension && udc.decoderSettings.Lossless == lossless && udc.decoderSettings.Name == name)
                {
                    result = udc;
                    return true;
                }
            }
            result = null;
            return false;
        }

        public AudioDecoderSettingsViewModel GetDefault(string extension, bool lossless)
        {
            AudioDecoderSettingsViewModel result = null;
            foreach (AudioDecoderSettingsViewModel udc in this)
            {
                if (udc.decoderSettings.Extension == extension && udc.decoderSettings.Lossless == lossless && (result == null || result.decoderSettings.Priority < udc.decoderSettings.Priority))
                {
                    result = udc;
                }
            }
            return result;
        }
    }
}
