using System;
using System.ComponentModel;

namespace CUETools.Codecs
{
    public class EncoderListViewModel : BindingList<AudioEncoderSettingsViewModel>
    {
        public EncoderListViewModel()
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
}
