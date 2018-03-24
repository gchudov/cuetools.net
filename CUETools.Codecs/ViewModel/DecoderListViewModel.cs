using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace CUETools.Codecs
{
    public class DecoderListViewModel : BindingList<AudioDecoderSettingsViewModel>
    {
        private List<AudioDecoderSettings> model;

        public DecoderListViewModel(List<AudioDecoderSettings> model)
            : base()
        {
            this.model = model;
            model.ForEach(item => Add(new AudioDecoderSettingsViewModel(item)));
            AddingNew += OnAddingNew;
        }

        private void OnAddingNew(object sender, AddingNewEventArgs e)
        {
            var item = new CommandLine.DecoderSettings("new", "wav", "", "");
            model.Add(item);
            e.NewObject = new AudioDecoderSettingsViewModel(item);
        }

        public bool TryGetValue(string extension, bool lossless, string name, out AudioDecoderSettingsViewModel result)
        {
            foreach (AudioDecoderSettingsViewModel udc in this)
            {
                if (udc.Settings.Extension == extension && udc.Settings.Lossless == lossless && udc.Settings.Name == name)
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
                if (udc.Settings.Extension == extension && udc.Settings.Lossless == lossless && (result == null || result.Settings.Priority < udc.Settings.Priority))
                {
                    result = udc;
                }
            }
            return result;
        }
    }
}
