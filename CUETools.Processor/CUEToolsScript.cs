using System.Collections.Generic;

namespace CUETools.Processor
{
    public class CUEToolsScript
    {
        public string name { get; set; }
        public bool builtin;
        public List<CUEAction> conditions;
        public string code;

        public CUEToolsScript(string _name, bool _builtin, IEnumerable<CUEAction> _conditions, string _code)
        {
            name = _name;
            builtin = _builtin;
            conditions = new List<CUEAction>();
            foreach (CUEAction condition in _conditions)
            {
                conditions.Add(condition);
            }
            code = _code;
        }

        public override string ToString()
        {
            return name;
        }
    }
}
