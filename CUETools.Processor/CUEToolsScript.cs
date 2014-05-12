using System.Collections.Generic;

namespace CUETools.Processor
{
    public class CUEToolsScript
    {
        public string name { get; set; }
        public List<CUEAction> conditions;

        public CUEToolsScript(string _name, IEnumerable<CUEAction> _conditions)
        {
            name = _name;
            conditions = new List<CUEAction>();
            foreach (CUEAction condition in _conditions)
            {
                conditions.Add(condition);
            }
        }

        public override string ToString()
        {
            return name;
        }
    }
}
