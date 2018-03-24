using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
    public class DecoderListViewModel : BindingList<AudioDecoderSettingsViewModel>
    {
        public DecoderListViewModel()
            : base()
        {
            AddingNew += OnAddingNew;
        }

        private void OnAddingNew(object sender, AddingNewEventArgs e)
        {
            e.NewObject = new AudioDecoderSettingsViewModel(new CommandLine.DecoderSettings("new", "wav", "", ""));
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
