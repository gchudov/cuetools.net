using System;
using System.ComponentModel;

namespace CUETools.Processor
{
    public class CUEToolsUDCList : BindingList<CUEToolsUDC>
    {
        public CUEToolsUDCList()
            : base()
        {
            AddingNew += OnAddingNew;
        }

        private void OnAddingNew(object sender, AddingNewEventArgs e)
        {
            e.NewObject = new CUEToolsUDC("new", "wav", true, "", "", "", "");
        }

        public bool TryGetValue(string extension, bool lossless, string name, out CUEToolsUDC result)
        {
            foreach (CUEToolsUDC udc in this)
            {
                if (udc.extension == extension && udc.lossless == lossless && udc.name == name)
                {
                    result = udc;
                    return true;
                }
            }
            result = null;
            return false;
        }

        public CUEToolsUDC GetDefault(string extension, bool lossless)
        {
            CUEToolsUDC result = null;
            foreach (CUEToolsUDC udc in this)
            {
                if (udc.extension == extension && udc.lossless == lossless && (result == null || result.priority < udc.priority))
                {
                    result = udc;
                }
            }
            return result;
        }
    }
}
