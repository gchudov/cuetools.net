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
            string name = "new";
            CUEToolsUDC temp;
            while (TryGetValue(name, out temp))
            {
                name += "(1)";
            }
            e.NewObject = new CUEToolsUDC(name, "wav", true, "", "", "", "");
        }

        public bool TryGetValue(string name, out CUEToolsUDC result)
        {
            foreach (CUEToolsUDC udc in this)
            {
                if (udc.name == name)
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

        public CUEToolsUDC this[string name]
        {
            get
            {
                CUEToolsUDC udc;
                if (!TryGetValue(name, out udc))
                {
                    throw new Exception("CUEToolsUDCList: member not found");
                }
                return udc;
            }
        }
    }
}
