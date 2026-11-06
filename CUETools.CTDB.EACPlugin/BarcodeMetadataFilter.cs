using System;
using System.Collections.Generic;
using CUETools.CTDB;

namespace CUETools.CTDB.EACPlugin
{
    internal static class BarcodeMetadataFilter
    {
        public static List<CTDBResponseMeta> FilterOrOriginal(IList<CTDBResponseMeta> metadata, string barcode)
        {
            List<CTDBResponseMeta> matches = new List<CTDBResponseMeta>();
            if (string.IsNullOrEmpty(barcode)) return new List<CTDBResponseMeta>(metadata);
            foreach (CTDBResponseMeta entry in metadata)
                if (entry != null && string.Equals(entry.barcode, barcode, StringComparison.Ordinal))
                    matches.Add(entry);
            return matches.Count == 0 ? new List<CTDBResponseMeta>(metadata) : matches;
        }

        public static bool HasSingleMatch(IList<CTDBResponseMeta> filtered, string barcode)
        {
            return !string.IsNullOrEmpty(barcode) && filtered.Count == 1 &&
                string.Equals(filtered[0].barcode, barcode, StringComparison.Ordinal);
        }

    }
}
