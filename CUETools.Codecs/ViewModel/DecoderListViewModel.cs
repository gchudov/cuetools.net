using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace CUETools.Codecs
{
    public class DecoderListViewModel : BindingList<AudioDecoderSettingsViewModel>
    {
        private List<IAudioDecoderSettings> model;

        public DecoderListViewModel(List<IAudioDecoderSettings> model)
            : base()
        {
            this.model = model;
            model.ForEach(item => Add(new AudioDecoderSettingsViewModel(item)));
            AddingNew += OnAddingNew;
            ListChanged += OnListChanged;
        }

        private void OnAddingNew(object sender, AddingNewEventArgs e)
        {
            var item = new CommandLine.DecoderSettings("new", "wav", "", "");
            model.Add(item);
            e.NewObject = new AudioDecoderSettingsViewModel(item);
        }

        private void OnListChanged(object sender, ListChangedEventArgs e)
        {
            if (e.ListChangedType == ListChangedType.ItemDeleted)
            {
                model.RemoveAt(e.NewIndex);
            }
        }

        public bool TryGetValue(string extension, string name, out AudioDecoderSettingsViewModel result)
        {
            foreach (AudioDecoderSettingsViewModel udc in this)
            {
                if (udc.Settings.Extension == extension && udc.Settings.Name == name)
                {
                    result = udc;
                    return true;
                }
            }
            result = null;
            return false;
        }

        public AudioDecoderSettingsViewModel GetDefault(string extension)
        {
            AudioDecoderSettingsViewModel result = null;
            foreach (AudioDecoderSettingsViewModel udc in this)
            {
                if (udc.Settings.Extension == extension && (result == null || result.Settings.Priority < udc.Settings.Priority))
                {
                    result = udc;
                }
            }
            return result;
        }
    }
}
