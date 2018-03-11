namespace Microsoft.Xml.Serialization.GeneratedAssembly {

    public class XmlSerializationWriter1 : System.Xml.Serialization.XmlSerializationWriter {

        public void Write15_ctdb(object o) {
            WriteStartDocument();
            if (o == null) {
                WriteNullTagLiteral(@"ctdb", @"http://db.cuetools.net/ns/mmd-1.0#");
                return;
            }
            TopLevelElement();
            Write8_CTDBResponse(@"ctdb", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::CUETools.CTDB.CTDBResponse)o), true, false);
        }

        public void Write16_CTDBResponseEntry(object o) {
            WriteStartDocument();
            if (o == null) {
                WriteNullTagLiteral(@"CTDBResponseEntry", @"");
                return;
            }
            TopLevelElement();
            Write9_CTDBResponseEntry(@"CTDBResponseEntry", @"", ((global::CUETools.CTDB.CTDBResponseEntry)o), true, false);
        }

        public void Write17_CTDBResponseMeta(object o) {
            WriteStartDocument();
            if (o == null) {
                WriteNullTagLiteral(@"CTDBResponseMeta", @"");
                return;
            }
            TopLevelElement();
            Write14_CTDBResponseMeta(@"CTDBResponseMeta", @"", ((global::CUETools.CTDB.CTDBResponseMeta)o), true, false);
        }

        public void Write18_CTDBResponseMetaImage(object o) {
            WriteStartDocument();
            if (o == null) {
                WriteNullTagLiteral(@"CTDBResponseMetaImage", @"");
                return;
            }
            TopLevelElement();
            Write10_CTDBResponseMetaImage(@"CTDBResponseMetaImage", @"", ((global::CUETools.CTDB.CTDBResponseMetaImage)o), true, false);
        }

        public void Write19_CTDBResponseMetaLabel(object o) {
            WriteStartDocument();
            if (o == null) {
                WriteNullTagLiteral(@"CTDBResponseMetaLabel", @"");
                return;
            }
            TopLevelElement();
            Write12_CTDBResponseMetaLabel(@"CTDBResponseMetaLabel", @"", ((global::CUETools.CTDB.CTDBResponseMetaLabel)o), true, false);
        }

        public void Write20_CTDBResponseMetaRelease(object o) {
            WriteStartDocument();
            if (o == null) {
                WriteNullTagLiteral(@"CTDBResponseMetaRelease", @"");
                return;
            }
            TopLevelElement();
            Write13_CTDBResponseMetaRelease(@"CTDBResponseMetaRelease", @"", ((global::CUETools.CTDB.CTDBResponseMetaRelease)o), true, false);
        }

        public void Write21_CTDBResponseMetaTrack(object o) {
            WriteStartDocument();
            if (o == null) {
                WriteNullTagLiteral(@"CTDBResponseMetaTrack", @"");
                return;
            }
            TopLevelElement();
            Write11_CTDBResponseMetaTrack(@"CTDBResponseMetaTrack", @"", ((global::CUETools.CTDB.CTDBResponseMetaTrack)o), true, false);
        }

        void Write11_CTDBResponseMetaTrack(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaTrack o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaTrack)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaTrack", @"");
            WriteAttribute(@"name", @"", ((global::System.String)o.@name));
            WriteAttribute(@"artist", @"", ((global::System.String)o.@artist));
            WriteElementString(@"extra", @"", ((global::System.String)o.@extra));
            WriteEndElement(o);
        }

        void Write13_CTDBResponseMetaRelease(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaRelease o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaRelease)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaRelease", @"");
            WriteAttribute(@"date", @"", ((global::System.String)o.@date));
            WriteAttribute(@"country", @"", ((global::System.String)o.@country));
            WriteEndElement(o);
        }

        void Write12_CTDBResponseMetaLabel(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaLabel o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaLabel)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaLabel", @"");
            WriteAttribute(@"name", @"", ((global::System.String)o.@name));
            WriteAttribute(@"catno", @"", ((global::System.String)o.@catno));
            WriteEndElement(o);
        }

        void Write10_CTDBResponseMetaImage(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaImage o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaImage)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaImage", @"");
            WriteAttribute(@"uri", @"", ((global::System.String)o.@uri));
            WriteAttribute(@"uri150", @"", ((global::System.String)o.@uri150));
            WriteAttribute(@"height", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@height)));
            WriteAttribute(@"width", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@width)));
            WriteAttribute(@"primary", @"", System.Xml.XmlConvert.ToString((global::System.Boolean)((global::System.Boolean)o.@primary)));
            WriteEndElement(o);
        }

        void Write14_CTDBResponseMeta(string n, string ns, global::CUETools.CTDB.CTDBResponseMeta o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMeta)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMeta", @"");
            WriteAttribute(@"source", @"", ((global::System.String)o.@source));
            WriteAttribute(@"id", @"", ((global::System.String)o.@id));
            WriteAttribute(@"artist", @"", ((global::System.String)o.@artist));
            WriteAttribute(@"album", @"", ((global::System.String)o.@album));
            WriteAttribute(@"year", @"", ((global::System.String)o.@year));
            WriteAttribute(@"genre", @"", ((global::System.String)o.@genre));
            WriteAttribute(@"discnumber", @"", ((global::System.String)o.@discnumber));
            WriteAttribute(@"disccount", @"", ((global::System.String)o.@disccount));
            WriteAttribute(@"discname", @"", ((global::System.String)o.@discname));
            WriteAttribute(@"infourl", @"", ((global::System.String)o.@infourl));
            WriteAttribute(@"barcode", @"", ((global::System.String)o.@barcode));
            {
                global::CUETools.CTDB.CTDBResponseMetaImage[] a = (global::CUETools.CTDB.CTDBResponseMetaImage[])o.@coverart;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write10_CTDBResponseMetaImage(@"coverart", @"", ((global::CUETools.CTDB.CTDBResponseMetaImage)a[ia]), false, false);
                    }
                }
            }
            {
                global::CUETools.CTDB.CTDBResponseMetaTrack[] a = (global::CUETools.CTDB.CTDBResponseMetaTrack[])o.@track;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write11_CTDBResponseMetaTrack(@"track", @"", ((global::CUETools.CTDB.CTDBResponseMetaTrack)a[ia]), false, false);
                    }
                }
            }
            {
                global::CUETools.CTDB.CTDBResponseMetaLabel[] a = (global::CUETools.CTDB.CTDBResponseMetaLabel[])o.@label;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write12_CTDBResponseMetaLabel(@"label", @"", ((global::CUETools.CTDB.CTDBResponseMetaLabel)a[ia]), false, false);
                    }
                }
            }
            {
                global::CUETools.CTDB.CTDBResponseMetaRelease[] a = (global::CUETools.CTDB.CTDBResponseMetaRelease[])o.@release;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write13_CTDBResponseMetaRelease(@"release", @"", ((global::CUETools.CTDB.CTDBResponseMetaRelease)a[ia]), false, false);
                    }
                }
            }
            WriteElementString(@"extra", @"", ((global::System.String)o.@extra));
            WriteEndElement(o);
        }

        void Write9_CTDBResponseEntry(string n, string ns, global::CUETools.CTDB.CTDBResponseEntry o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseEntry)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseEntry", @"");
            WriteAttribute(@"id", @"", System.Xml.XmlConvert.ToString((global::System.Int64)((global::System.Int64)o.@id)));
            WriteAttribute(@"crc32", @"", ((global::System.String)o.@crc32));
            WriteAttribute(@"confidence", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@confidence)));
            WriteAttribute(@"npar", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@npar)));
            WriteAttribute(@"stride", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@stride)));
            WriteAttribute(@"hasparity", @"", ((global::System.String)o.@hasparity));
            WriteAttribute(@"parity", @"", ((global::System.String)o.@parity));
            WriteAttribute(@"syndrome", @"", ((global::System.String)o.@syndrome));
            WriteAttribute(@"trackcrcs", @"", ((global::System.String)o.@trackcrcs));
            WriteAttribute(@"toc", @"", ((global::System.String)o.@toc));
            WriteEndElement(o);
        }

        void Write8_CTDBResponse(string n, string ns, global::CUETools.CTDB.CTDBResponse o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponse)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponse", @"http://db.cuetools.net/ns/mmd-1.0#");
            WriteAttribute(@"status", @"", ((global::System.String)o.@status));
            WriteAttribute(@"updateurl", @"", ((global::System.String)o.@updateurl));
            WriteAttribute(@"updatemsg", @"", ((global::System.String)o.@updatemsg));
            WriteAttribute(@"message", @"", ((global::System.String)o.@message));
            WriteAttribute(@"npar", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@npar)));
            {
                global::CUETools.CTDB.CTDBResponseEntry[] a = (global::CUETools.CTDB.CTDBResponseEntry[])o.@entry;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write2_CTDBResponseEntry(@"entry", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::CUETools.CTDB.CTDBResponseEntry)a[ia]), false, false);
                    }
                }
            }
            {
                global::CUETools.CTDB.CTDBResponseMeta[] a = (global::CUETools.CTDB.CTDBResponseMeta[])o.@metadata;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write7_CTDBResponseMeta(@"metadata", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::CUETools.CTDB.CTDBResponseMeta)a[ia]), false, false);
                    }
                }
            }
            WriteEndElement(o);
        }

        void Write7_CTDBResponseMeta(string n, string ns, global::CUETools.CTDB.CTDBResponseMeta o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMeta)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMeta", @"http://db.cuetools.net/ns/mmd-1.0#");
            WriteAttribute(@"source", @"", ((global::System.String)o.@source));
            WriteAttribute(@"id", @"", ((global::System.String)o.@id));
            WriteAttribute(@"artist", @"", ((global::System.String)o.@artist));
            WriteAttribute(@"album", @"", ((global::System.String)o.@album));
            WriteAttribute(@"year", @"", ((global::System.String)o.@year));
            WriteAttribute(@"genre", @"", ((global::System.String)o.@genre));
            WriteAttribute(@"discnumber", @"", ((global::System.String)o.@discnumber));
            WriteAttribute(@"disccount", @"", ((global::System.String)o.@disccount));
            WriteAttribute(@"discname", @"", ((global::System.String)o.@discname));
            WriteAttribute(@"infourl", @"", ((global::System.String)o.@infourl));
            WriteAttribute(@"barcode", @"", ((global::System.String)o.@barcode));
            {
                global::CUETools.CTDB.CTDBResponseMetaImage[] a = (global::CUETools.CTDB.CTDBResponseMetaImage[])o.@coverart;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write3_CTDBResponseMetaImage(@"coverart", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::CUETools.CTDB.CTDBResponseMetaImage)a[ia]), false, false);
                    }
                }
            }
            {
                global::CUETools.CTDB.CTDBResponseMetaTrack[] a = (global::CUETools.CTDB.CTDBResponseMetaTrack[])o.@track;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write4_CTDBResponseMetaTrack(@"track", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::CUETools.CTDB.CTDBResponseMetaTrack)a[ia]), false, false);
                    }
                }
            }
            {
                global::CUETools.CTDB.CTDBResponseMetaLabel[] a = (global::CUETools.CTDB.CTDBResponseMetaLabel[])o.@label;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write5_CTDBResponseMetaLabel(@"label", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::CUETools.CTDB.CTDBResponseMetaLabel)a[ia]), false, false);
                    }
                }
            }
            {
                global::CUETools.CTDB.CTDBResponseMetaRelease[] a = (global::CUETools.CTDB.CTDBResponseMetaRelease[])o.@release;
                if (a != null) {
                    for (int ia = 0; ia < a.Length; ia++) {
                        Write6_CTDBResponseMetaRelease(@"release", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::CUETools.CTDB.CTDBResponseMetaRelease)a[ia]), false, false);
                    }
                }
            }
            WriteElementString(@"extra", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::System.String)o.@extra));
            WriteEndElement(o);
        }

        void Write6_CTDBResponseMetaRelease(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaRelease o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaRelease)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaRelease", @"http://db.cuetools.net/ns/mmd-1.0#");
            WriteAttribute(@"date", @"", ((global::System.String)o.@date));
            WriteAttribute(@"country", @"", ((global::System.String)o.@country));
            WriteEndElement(o);
        }

        void Write5_CTDBResponseMetaLabel(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaLabel o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaLabel)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaLabel", @"http://db.cuetools.net/ns/mmd-1.0#");
            WriteAttribute(@"name", @"", ((global::System.String)o.@name));
            WriteAttribute(@"catno", @"", ((global::System.String)o.@catno));
            WriteEndElement(o);
        }

        void Write4_CTDBResponseMetaTrack(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaTrack o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaTrack)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaTrack", @"http://db.cuetools.net/ns/mmd-1.0#");
            WriteAttribute(@"name", @"", ((global::System.String)o.@name));
            WriteAttribute(@"artist", @"", ((global::System.String)o.@artist));
            WriteElementString(@"extra", @"http://db.cuetools.net/ns/mmd-1.0#", ((global::System.String)o.@extra));
            WriteEndElement(o);
        }

        void Write3_CTDBResponseMetaImage(string n, string ns, global::CUETools.CTDB.CTDBResponseMetaImage o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseMetaImage)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseMetaImage", @"http://db.cuetools.net/ns/mmd-1.0#");
            WriteAttribute(@"uri", @"", ((global::System.String)o.@uri));
            WriteAttribute(@"uri150", @"", ((global::System.String)o.@uri150));
            WriteAttribute(@"height", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@height)));
            WriteAttribute(@"width", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@width)));
            WriteAttribute(@"primary", @"", System.Xml.XmlConvert.ToString((global::System.Boolean)((global::System.Boolean)o.@primary)));
            WriteEndElement(o);
        }

        void Write2_CTDBResponseEntry(string n, string ns, global::CUETools.CTDB.CTDBResponseEntry o, bool isNullable, bool needType) {
            if ((object)o == null) {
                if (isNullable) WriteNullTagLiteral(n, ns);
                return;
            }
            if (!needType) {
                System.Type t = o.GetType();
                if (t == typeof(global::CUETools.CTDB.CTDBResponseEntry)) {
                }
                else {
                    throw CreateUnknownTypeException(o);
                }
            }
            WriteStartElement(n, ns, o, false, null);
            if (needType) WriteXsiType(@"CTDBResponseEntry", @"http://db.cuetools.net/ns/mmd-1.0#");
            WriteAttribute(@"id", @"", System.Xml.XmlConvert.ToString((global::System.Int64)((global::System.Int64)o.@id)));
            WriteAttribute(@"crc32", @"", ((global::System.String)o.@crc32));
            WriteAttribute(@"confidence", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@confidence)));
            WriteAttribute(@"npar", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@npar)));
            WriteAttribute(@"stride", @"", System.Xml.XmlConvert.ToString((global::System.Int32)((global::System.Int32)o.@stride)));
            WriteAttribute(@"hasparity", @"", ((global::System.String)o.@hasparity));
            WriteAttribute(@"parity", @"", ((global::System.String)o.@parity));
            WriteAttribute(@"syndrome", @"", ((global::System.String)o.@syndrome));
            WriteAttribute(@"trackcrcs", @"", ((global::System.String)o.@trackcrcs));
            WriteAttribute(@"toc", @"", ((global::System.String)o.@toc));
            WriteEndElement(o);
        }

        protected override void InitCallbacks() {
        }
    }

    public class XmlSerializationReader1 : System.Xml.Serialization.XmlSerializationReader {

        public object Read15_ctdb() {
            object o = null;
            Reader.MoveToContent();
            if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                if (((object) Reader.LocalName == (object)id1_ctdb && (object) Reader.NamespaceURI == (object)id2_Item)) {
                    o = Read8_CTDBResponse(true, true);
                }
                else {
                    throw CreateUnknownNodeException();
                }
            }
            else {
                UnknownNode(null, @"http://db.cuetools.net/ns/mmd-1.0#:ctdb");
            }
            return (object)o;
        }

        public object Read16_CTDBResponseEntry() {
            object o = null;
            Reader.MoveToContent();
            if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                if (((object) Reader.LocalName == (object)id3_CTDBResponseEntry && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o = Read9_CTDBResponseEntry(true, true);
                }
                else {
                    throw CreateUnknownNodeException();
                }
            }
            else {
                UnknownNode(null, @":CTDBResponseEntry");
            }
            return (object)o;
        }

        public object Read17_CTDBResponseMeta() {
            object o = null;
            Reader.MoveToContent();
            if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                if (((object) Reader.LocalName == (object)id5_CTDBResponseMeta && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o = Read14_CTDBResponseMeta(true, true);
                }
                else {
                    throw CreateUnknownNodeException();
                }
            }
            else {
                UnknownNode(null, @":CTDBResponseMeta");
            }
            return (object)o;
        }

        public object Read18_CTDBResponseMetaImage() {
            object o = null;
            Reader.MoveToContent();
            if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                if (((object) Reader.LocalName == (object)id6_CTDBResponseMetaImage && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o = Read10_CTDBResponseMetaImage(true, true);
                }
                else {
                    throw CreateUnknownNodeException();
                }
            }
            else {
                UnknownNode(null, @":CTDBResponseMetaImage");
            }
            return (object)o;
        }

        public object Read19_CTDBResponseMetaLabel() {
            object o = null;
            Reader.MoveToContent();
            if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                if (((object) Reader.LocalName == (object)id7_CTDBResponseMetaLabel && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o = Read12_CTDBResponseMetaLabel(true, true);
                }
                else {
                    throw CreateUnknownNodeException();
                }
            }
            else {
                UnknownNode(null, @":CTDBResponseMetaLabel");
            }
            return (object)o;
        }

        public object Read20_CTDBResponseMetaRelease() {
            object o = null;
            Reader.MoveToContent();
            if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                if (((object) Reader.LocalName == (object)id8_CTDBResponseMetaRelease && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o = Read13_CTDBResponseMetaRelease(true, true);
                }
                else {
                    throw CreateUnknownNodeException();
                }
            }
            else {
                UnknownNode(null, @":CTDBResponseMetaRelease");
            }
            return (object)o;
        }

        public object Read21_CTDBResponseMetaTrack() {
            object o = null;
            Reader.MoveToContent();
            if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                if (((object) Reader.LocalName == (object)id9_CTDBResponseMetaTrack && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o = Read11_CTDBResponseMetaTrack(true, true);
                }
                else {
                    throw CreateUnknownNodeException();
                }
            }
            else {
                UnknownNode(null, @":CTDBResponseMetaTrack");
            }
            return (object)o;
        }

        global::CUETools.CTDB.CTDBResponseMetaTrack Read11_CTDBResponseMetaTrack(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id9_CTDBResponseMetaTrack && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id4_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaTrack o;
            o = new global::CUETools.CTDB.CTDBResponseMetaTrack();
            bool[] paramsRead = new bool[3];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id10_name && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@name = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id11_artist && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@artist = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":name, :artist");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations0 = 0;
            int readerCount0 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    if (!paramsRead[2] && ((object) Reader.LocalName == (object)id12_extra && (object) Reader.NamespaceURI == (object)id4_Item)) {
                        {
                            o.@extra = Reader.ReadElementString();
                        }
                        paramsRead[2] = true;
                    }
                    else {
                        UnknownNode((object)o, @":extra");
                    }
                }
                else {
                    UnknownNode((object)o, @":extra");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations0, ref readerCount0);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMetaRelease Read13_CTDBResponseMetaRelease(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id8_CTDBResponseMetaRelease && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id4_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaRelease o;
            o = new global::CUETools.CTDB.CTDBResponseMetaRelease();
            bool[] paramsRead = new bool[2];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id13_date && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@date = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id14_country && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@country = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":date, :country");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations1 = 0;
            int readerCount1 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations1, ref readerCount1);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMetaLabel Read12_CTDBResponseMetaLabel(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id7_CTDBResponseMetaLabel && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id4_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaLabel o;
            o = new global::CUETools.CTDB.CTDBResponseMetaLabel();
            bool[] paramsRead = new bool[2];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id10_name && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@name = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id15_catno && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@catno = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":name, :catno");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations2 = 0;
            int readerCount2 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations2, ref readerCount2);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMetaImage Read10_CTDBResponseMetaImage(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id6_CTDBResponseMetaImage && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id4_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaImage o;
            o = new global::CUETools.CTDB.CTDBResponseMetaImage();
            bool[] paramsRead = new bool[5];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id16_uri && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@uri = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id17_uri150 && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@uri150 = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!paramsRead[2] && ((object) Reader.LocalName == (object)id18_height && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@height = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[2] = true;
                }
                else if (!paramsRead[3] && ((object) Reader.LocalName == (object)id19_width && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@width = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[3] = true;
                }
                else if (!paramsRead[4] && ((object) Reader.LocalName == (object)id20_primary && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@primary = System.Xml.XmlConvert.ToBoolean(Reader.Value);
                    paramsRead[4] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":uri, :uri150, :height, :width, :primary");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations3 = 0;
            int readerCount3 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations3, ref readerCount3);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMeta Read14_CTDBResponseMeta(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id5_CTDBResponseMeta && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id4_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMeta o;
            o = new global::CUETools.CTDB.CTDBResponseMeta();
            global::CUETools.CTDB.CTDBResponseMetaImage[] a_0 = null;
            int ca_0 = 0;
            global::CUETools.CTDB.CTDBResponseMetaTrack[] a_1 = null;
            int ca_1 = 0;
            global::CUETools.CTDB.CTDBResponseMetaLabel[] a_2 = null;
            int ca_2 = 0;
            global::CUETools.CTDB.CTDBResponseMetaRelease[] a_3 = null;
            int ca_3 = 0;
            bool[] paramsRead = new bool[16];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[4] && ((object) Reader.LocalName == (object)id21_source && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@source = Reader.Value;
                    paramsRead[4] = true;
                }
                else if (!paramsRead[5] && ((object) Reader.LocalName == (object)id22_id && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@id = Reader.Value;
                    paramsRead[5] = true;
                }
                else if (!paramsRead[6] && ((object) Reader.LocalName == (object)id11_artist && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@artist = Reader.Value;
                    paramsRead[6] = true;
                }
                else if (!paramsRead[7] && ((object) Reader.LocalName == (object)id23_album && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@album = Reader.Value;
                    paramsRead[7] = true;
                }
                else if (!paramsRead[8] && ((object) Reader.LocalName == (object)id24_year && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@year = Reader.Value;
                    paramsRead[8] = true;
                }
                else if (!paramsRead[9] && ((object) Reader.LocalName == (object)id25_genre && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@genre = Reader.Value;
                    paramsRead[9] = true;
                }
                else if (!paramsRead[11] && ((object) Reader.LocalName == (object)id26_discnumber && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@discnumber = Reader.Value;
                    paramsRead[11] = true;
                }
                else if (!paramsRead[12] && ((object) Reader.LocalName == (object)id27_disccount && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@disccount = Reader.Value;
                    paramsRead[12] = true;
                }
                else if (!paramsRead[13] && ((object) Reader.LocalName == (object)id28_discname && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@discname = Reader.Value;
                    paramsRead[13] = true;
                }
                else if (!paramsRead[14] && ((object) Reader.LocalName == (object)id29_infourl && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@infourl = Reader.Value;
                    paramsRead[14] = true;
                }
                else if (!paramsRead[15] && ((object) Reader.LocalName == (object)id30_barcode && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@barcode = Reader.Value;
                    paramsRead[15] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":source, :id, :artist, :album, :year, :genre, :discnumber, :disccount, :discname, :infourl, :barcode");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                o.@coverart = (global::CUETools.CTDB.CTDBResponseMetaImage[])ShrinkArray(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseMetaImage), true);
                o.@track = (global::CUETools.CTDB.CTDBResponseMetaTrack[])ShrinkArray(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMetaTrack), true);
                o.@label = (global::CUETools.CTDB.CTDBResponseMetaLabel[])ShrinkArray(a_2, ca_2, typeof(global::CUETools.CTDB.CTDBResponseMetaLabel), true);
                o.@release = (global::CUETools.CTDB.CTDBResponseMetaRelease[])ShrinkArray(a_3, ca_3, typeof(global::CUETools.CTDB.CTDBResponseMetaRelease), true);
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations4 = 0;
            int readerCount4 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    if (((object) Reader.LocalName == (object)id31_coverart && (object) Reader.NamespaceURI == (object)id4_Item)) {
                        a_0 = (global::CUETools.CTDB.CTDBResponseMetaImage[])EnsureArrayIndex(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseMetaImage));a_0[ca_0++] = Read10_CTDBResponseMetaImage(false, true);
                    }
                    else if (((object) Reader.LocalName == (object)id32_track && (object) Reader.NamespaceURI == (object)id4_Item)) {
                        a_1 = (global::CUETools.CTDB.CTDBResponseMetaTrack[])EnsureArrayIndex(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMetaTrack));a_1[ca_1++] = Read11_CTDBResponseMetaTrack(false, true);
                    }
                    else if (((object) Reader.LocalName == (object)id33_label && (object) Reader.NamespaceURI == (object)id4_Item)) {
                        a_2 = (global::CUETools.CTDB.CTDBResponseMetaLabel[])EnsureArrayIndex(a_2, ca_2, typeof(global::CUETools.CTDB.CTDBResponseMetaLabel));a_2[ca_2++] = Read12_CTDBResponseMetaLabel(false, true);
                    }
                    else if (((object) Reader.LocalName == (object)id34_release && (object) Reader.NamespaceURI == (object)id4_Item)) {
                        a_3 = (global::CUETools.CTDB.CTDBResponseMetaRelease[])EnsureArrayIndex(a_3, ca_3, typeof(global::CUETools.CTDB.CTDBResponseMetaRelease));a_3[ca_3++] = Read13_CTDBResponseMetaRelease(false, true);
                    }
                    else if (!paramsRead[10] && ((object) Reader.LocalName == (object)id12_extra && (object) Reader.NamespaceURI == (object)id4_Item)) {
                        {
                            o.@extra = Reader.ReadElementString();
                        }
                        paramsRead[10] = true;
                    }
                    else {
                        UnknownNode((object)o, @":coverart, :track, :label, :release, :extra");
                    }
                }
                else {
                    UnknownNode((object)o, @":coverart, :track, :label, :release, :extra");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations4, ref readerCount4);
            }
            o.@coverart = (global::CUETools.CTDB.CTDBResponseMetaImage[])ShrinkArray(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseMetaImage), true);
            o.@track = (global::CUETools.CTDB.CTDBResponseMetaTrack[])ShrinkArray(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMetaTrack), true);
            o.@label = (global::CUETools.CTDB.CTDBResponseMetaLabel[])ShrinkArray(a_2, ca_2, typeof(global::CUETools.CTDB.CTDBResponseMetaLabel), true);
            o.@release = (global::CUETools.CTDB.CTDBResponseMetaRelease[])ShrinkArray(a_3, ca_3, typeof(global::CUETools.CTDB.CTDBResponseMetaRelease), true);
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseEntry Read9_CTDBResponseEntry(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id3_CTDBResponseEntry && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id4_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseEntry o;
            o = new global::CUETools.CTDB.CTDBResponseEntry();
            bool[] paramsRead = new bool[10];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id22_id && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@id = System.Xml.XmlConvert.ToInt64(Reader.Value);
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id35_crc32 && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@crc32 = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!paramsRead[2] && ((object) Reader.LocalName == (object)id36_confidence && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@confidence = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[2] = true;
                }
                else if (!paramsRead[3] && ((object) Reader.LocalName == (object)id37_npar && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@npar = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[3] = true;
                }
                else if (!paramsRead[4] && ((object) Reader.LocalName == (object)id38_stride && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@stride = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[4] = true;
                }
                else if (!paramsRead[5] && ((object) Reader.LocalName == (object)id39_hasparity && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@hasparity = Reader.Value;
                    paramsRead[5] = true;
                }
                else if (!paramsRead[6] && ((object) Reader.LocalName == (object)id40_parity && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@parity = Reader.Value;
                    paramsRead[6] = true;
                }
                else if (!paramsRead[7] && ((object) Reader.LocalName == (object)id41_syndrome && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@syndrome = Reader.Value;
                    paramsRead[7] = true;
                }
                else if (!paramsRead[8] && ((object) Reader.LocalName == (object)id42_trackcrcs && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@trackcrcs = Reader.Value;
                    paramsRead[8] = true;
                }
                else if (!paramsRead[9] && ((object) Reader.LocalName == (object)id43_toc && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@toc = Reader.Value;
                    paramsRead[9] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":id, :crc32, :confidence, :npar, :stride, :hasparity, :parity, :syndrome, :trackcrcs, :toc");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations5 = 0;
            int readerCount5 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations5, ref readerCount5);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponse Read8_CTDBResponse(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id44_CTDBResponse && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id2_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponse o;
            o = new global::CUETools.CTDB.CTDBResponse();
            global::CUETools.CTDB.CTDBResponseEntry[] a_0 = null;
            int ca_0 = 0;
            global::CUETools.CTDB.CTDBResponseMeta[] a_1 = null;
            int ca_1 = 0;
            bool[] paramsRead = new bool[7];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[2] && ((object) Reader.LocalName == (object)id45_status && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@status = Reader.Value;
                    paramsRead[2] = true;
                }
                else if (!paramsRead[3] && ((object) Reader.LocalName == (object)id46_updateurl && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@updateurl = Reader.Value;
                    paramsRead[3] = true;
                }
                else if (!paramsRead[4] && ((object) Reader.LocalName == (object)id47_updatemsg && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@updatemsg = Reader.Value;
                    paramsRead[4] = true;
                }
                else if (!paramsRead[5] && ((object) Reader.LocalName == (object)id48_message && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@message = Reader.Value;
                    paramsRead[5] = true;
                }
                else if (!paramsRead[6] && ((object) Reader.LocalName == (object)id37_npar && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@npar = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[6] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":status, :updateurl, :updatemsg, :message, :npar");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                o.@entry = (global::CUETools.CTDB.CTDBResponseEntry[])ShrinkArray(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseEntry), true);
                o.@metadata = (global::CUETools.CTDB.CTDBResponseMeta[])ShrinkArray(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMeta), true);
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations6 = 0;
            int readerCount6 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    if (((object) Reader.LocalName == (object)id49_entry && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        a_0 = (global::CUETools.CTDB.CTDBResponseEntry[])EnsureArrayIndex(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseEntry));a_0[ca_0++] = Read2_CTDBResponseEntry(false, true);
                    }
                    else if (((object) Reader.LocalName == (object)id50_metadata && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        a_1 = (global::CUETools.CTDB.CTDBResponseMeta[])EnsureArrayIndex(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMeta));a_1[ca_1++] = Read7_CTDBResponseMeta(false, true);
                    }
                    else {
                        UnknownNode((object)o, @"http://db.cuetools.net/ns/mmd-1.0#:entry, http://db.cuetools.net/ns/mmd-1.0#:metadata");
                    }
                }
                else {
                    UnknownNode((object)o, @"http://db.cuetools.net/ns/mmd-1.0#:entry, http://db.cuetools.net/ns/mmd-1.0#:metadata");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations6, ref readerCount6);
            }
            o.@entry = (global::CUETools.CTDB.CTDBResponseEntry[])ShrinkArray(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseEntry), true);
            o.@metadata = (global::CUETools.CTDB.CTDBResponseMeta[])ShrinkArray(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMeta), true);
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMeta Read7_CTDBResponseMeta(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id5_CTDBResponseMeta && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id2_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMeta o;
            o = new global::CUETools.CTDB.CTDBResponseMeta();
            global::CUETools.CTDB.CTDBResponseMetaImage[] a_0 = null;
            int ca_0 = 0;
            global::CUETools.CTDB.CTDBResponseMetaTrack[] a_1 = null;
            int ca_1 = 0;
            global::CUETools.CTDB.CTDBResponseMetaLabel[] a_2 = null;
            int ca_2 = 0;
            global::CUETools.CTDB.CTDBResponseMetaRelease[] a_3 = null;
            int ca_3 = 0;
            bool[] paramsRead = new bool[16];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[4] && ((object) Reader.LocalName == (object)id21_source && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@source = Reader.Value;
                    paramsRead[4] = true;
                }
                else if (!paramsRead[5] && ((object) Reader.LocalName == (object)id22_id && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@id = Reader.Value;
                    paramsRead[5] = true;
                }
                else if (!paramsRead[6] && ((object) Reader.LocalName == (object)id11_artist && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@artist = Reader.Value;
                    paramsRead[6] = true;
                }
                else if (!paramsRead[7] && ((object) Reader.LocalName == (object)id23_album && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@album = Reader.Value;
                    paramsRead[7] = true;
                }
                else if (!paramsRead[8] && ((object) Reader.LocalName == (object)id24_year && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@year = Reader.Value;
                    paramsRead[8] = true;
                }
                else if (!paramsRead[9] && ((object) Reader.LocalName == (object)id25_genre && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@genre = Reader.Value;
                    paramsRead[9] = true;
                }
                else if (!paramsRead[11] && ((object) Reader.LocalName == (object)id26_discnumber && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@discnumber = Reader.Value;
                    paramsRead[11] = true;
                }
                else if (!paramsRead[12] && ((object) Reader.LocalName == (object)id27_disccount && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@disccount = Reader.Value;
                    paramsRead[12] = true;
                }
                else if (!paramsRead[13] && ((object) Reader.LocalName == (object)id28_discname && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@discname = Reader.Value;
                    paramsRead[13] = true;
                }
                else if (!paramsRead[14] && ((object) Reader.LocalName == (object)id29_infourl && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@infourl = Reader.Value;
                    paramsRead[14] = true;
                }
                else if (!paramsRead[15] && ((object) Reader.LocalName == (object)id30_barcode && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@barcode = Reader.Value;
                    paramsRead[15] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":source, :id, :artist, :album, :year, :genre, :discnumber, :disccount, :discname, :infourl, :barcode");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                o.@coverart = (global::CUETools.CTDB.CTDBResponseMetaImage[])ShrinkArray(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseMetaImage), true);
                o.@track = (global::CUETools.CTDB.CTDBResponseMetaTrack[])ShrinkArray(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMetaTrack), true);
                o.@label = (global::CUETools.CTDB.CTDBResponseMetaLabel[])ShrinkArray(a_2, ca_2, typeof(global::CUETools.CTDB.CTDBResponseMetaLabel), true);
                o.@release = (global::CUETools.CTDB.CTDBResponseMetaRelease[])ShrinkArray(a_3, ca_3, typeof(global::CUETools.CTDB.CTDBResponseMetaRelease), true);
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations7 = 0;
            int readerCount7 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    if (((object) Reader.LocalName == (object)id31_coverart && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        a_0 = (global::CUETools.CTDB.CTDBResponseMetaImage[])EnsureArrayIndex(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseMetaImage));a_0[ca_0++] = Read3_CTDBResponseMetaImage(false, true);
                    }
                    else if (((object) Reader.LocalName == (object)id32_track && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        a_1 = (global::CUETools.CTDB.CTDBResponseMetaTrack[])EnsureArrayIndex(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMetaTrack));a_1[ca_1++] = Read4_CTDBResponseMetaTrack(false, true);
                    }
                    else if (((object) Reader.LocalName == (object)id33_label && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        a_2 = (global::CUETools.CTDB.CTDBResponseMetaLabel[])EnsureArrayIndex(a_2, ca_2, typeof(global::CUETools.CTDB.CTDBResponseMetaLabel));a_2[ca_2++] = Read5_CTDBResponseMetaLabel(false, true);
                    }
                    else if (((object) Reader.LocalName == (object)id34_release && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        a_3 = (global::CUETools.CTDB.CTDBResponseMetaRelease[])EnsureArrayIndex(a_3, ca_3, typeof(global::CUETools.CTDB.CTDBResponseMetaRelease));a_3[ca_3++] = Read6_CTDBResponseMetaRelease(false, true);
                    }
                    else if (!paramsRead[10] && ((object) Reader.LocalName == (object)id12_extra && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        {
                            o.@extra = Reader.ReadElementString();
                        }
                        paramsRead[10] = true;
                    }
                    else {
                        UnknownNode((object)o, @"http://db.cuetools.net/ns/mmd-1.0#:coverart, http://db.cuetools.net/ns/mmd-1.0#:track, http://db.cuetools.net/ns/mmd-1.0#:label, http://db.cuetools.net/ns/mmd-1.0#:release, http://db.cuetools.net/ns/mmd-1.0#:extra");
                    }
                }
                else {
                    UnknownNode((object)o, @"http://db.cuetools.net/ns/mmd-1.0#:coverart, http://db.cuetools.net/ns/mmd-1.0#:track, http://db.cuetools.net/ns/mmd-1.0#:label, http://db.cuetools.net/ns/mmd-1.0#:release, http://db.cuetools.net/ns/mmd-1.0#:extra");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations7, ref readerCount7);
            }
            o.@coverart = (global::CUETools.CTDB.CTDBResponseMetaImage[])ShrinkArray(a_0, ca_0, typeof(global::CUETools.CTDB.CTDBResponseMetaImage), true);
            o.@track = (global::CUETools.CTDB.CTDBResponseMetaTrack[])ShrinkArray(a_1, ca_1, typeof(global::CUETools.CTDB.CTDBResponseMetaTrack), true);
            o.@label = (global::CUETools.CTDB.CTDBResponseMetaLabel[])ShrinkArray(a_2, ca_2, typeof(global::CUETools.CTDB.CTDBResponseMetaLabel), true);
            o.@release = (global::CUETools.CTDB.CTDBResponseMetaRelease[])ShrinkArray(a_3, ca_3, typeof(global::CUETools.CTDB.CTDBResponseMetaRelease), true);
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMetaRelease Read6_CTDBResponseMetaRelease(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id8_CTDBResponseMetaRelease && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id2_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaRelease o;
            o = new global::CUETools.CTDB.CTDBResponseMetaRelease();
            bool[] paramsRead = new bool[2];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id13_date && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@date = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id14_country && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@country = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":date, :country");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations8 = 0;
            int readerCount8 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations8, ref readerCount8);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMetaLabel Read5_CTDBResponseMetaLabel(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id7_CTDBResponseMetaLabel && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id2_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaLabel o;
            o = new global::CUETools.CTDB.CTDBResponseMetaLabel();
            bool[] paramsRead = new bool[2];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id10_name && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@name = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id15_catno && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@catno = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":name, :catno");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations9 = 0;
            int readerCount9 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations9, ref readerCount9);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMetaTrack Read4_CTDBResponseMetaTrack(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id9_CTDBResponseMetaTrack && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id2_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaTrack o;
            o = new global::CUETools.CTDB.CTDBResponseMetaTrack();
            bool[] paramsRead = new bool[3];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id10_name && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@name = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id11_artist && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@artist = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":name, :artist");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations10 = 0;
            int readerCount10 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    if (!paramsRead[2] && ((object) Reader.LocalName == (object)id12_extra && (object) Reader.NamespaceURI == (object)id2_Item)) {
                        {
                            o.@extra = Reader.ReadElementString();
                        }
                        paramsRead[2] = true;
                    }
                    else {
                        UnknownNode((object)o, @"http://db.cuetools.net/ns/mmd-1.0#:extra");
                    }
                }
                else {
                    UnknownNode((object)o, @"http://db.cuetools.net/ns/mmd-1.0#:extra");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations10, ref readerCount10);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseMetaImage Read3_CTDBResponseMetaImage(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id6_CTDBResponseMetaImage && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id2_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseMetaImage o;
            o = new global::CUETools.CTDB.CTDBResponseMetaImage();
            bool[] paramsRead = new bool[5];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id16_uri && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@uri = Reader.Value;
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id17_uri150 && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@uri150 = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!paramsRead[2] && ((object) Reader.LocalName == (object)id18_height && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@height = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[2] = true;
                }
                else if (!paramsRead[3] && ((object) Reader.LocalName == (object)id19_width && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@width = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[3] = true;
                }
                else if (!paramsRead[4] && ((object) Reader.LocalName == (object)id20_primary && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@primary = System.Xml.XmlConvert.ToBoolean(Reader.Value);
                    paramsRead[4] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":uri, :uri150, :height, :width, :primary");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations11 = 0;
            int readerCount11 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations11, ref readerCount11);
            }
            ReadEndElement();
            return o;
        }

        global::CUETools.CTDB.CTDBResponseEntry Read2_CTDBResponseEntry(bool isNullable, bool checkType) {
            System.Xml.XmlQualifiedName xsiType = checkType ? GetXsiType() : null;
            bool isNull = false;
            if (isNullable) isNull = ReadNull();
            if (checkType) {
            if (xsiType == null || ((object) ((System.Xml.XmlQualifiedName)xsiType).Name == (object)id3_CTDBResponseEntry && (object) ((System.Xml.XmlQualifiedName)xsiType).Namespace == (object)id2_Item)) {
            }
            else
                throw CreateUnknownTypeException((System.Xml.XmlQualifiedName)xsiType);
            }
            if (isNull) return null;
            global::CUETools.CTDB.CTDBResponseEntry o;
            o = new global::CUETools.CTDB.CTDBResponseEntry();
            bool[] paramsRead = new bool[10];
            while (Reader.MoveToNextAttribute()) {
                if (!paramsRead[0] && ((object) Reader.LocalName == (object)id22_id && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@id = System.Xml.XmlConvert.ToInt64(Reader.Value);
                    paramsRead[0] = true;
                }
                else if (!paramsRead[1] && ((object) Reader.LocalName == (object)id35_crc32 && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@crc32 = Reader.Value;
                    paramsRead[1] = true;
                }
                else if (!paramsRead[2] && ((object) Reader.LocalName == (object)id36_confidence && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@confidence = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[2] = true;
                }
                else if (!paramsRead[3] && ((object) Reader.LocalName == (object)id37_npar && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@npar = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[3] = true;
                }
                else if (!paramsRead[4] && ((object) Reader.LocalName == (object)id38_stride && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@stride = System.Xml.XmlConvert.ToInt32(Reader.Value);
                    paramsRead[4] = true;
                }
                else if (!paramsRead[5] && ((object) Reader.LocalName == (object)id39_hasparity && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@hasparity = Reader.Value;
                    paramsRead[5] = true;
                }
                else if (!paramsRead[6] && ((object) Reader.LocalName == (object)id40_parity && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@parity = Reader.Value;
                    paramsRead[6] = true;
                }
                else if (!paramsRead[7] && ((object) Reader.LocalName == (object)id41_syndrome && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@syndrome = Reader.Value;
                    paramsRead[7] = true;
                }
                else if (!paramsRead[8] && ((object) Reader.LocalName == (object)id42_trackcrcs && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@trackcrcs = Reader.Value;
                    paramsRead[8] = true;
                }
                else if (!paramsRead[9] && ((object) Reader.LocalName == (object)id43_toc && (object) Reader.NamespaceURI == (object)id4_Item)) {
                    o.@toc = Reader.Value;
                    paramsRead[9] = true;
                }
                else if (!IsXmlnsAttribute(Reader.Name)) {
                    UnknownNode((object)o, @":id, :crc32, :confidence, :npar, :stride, :hasparity, :parity, :syndrome, :trackcrcs, :toc");
                }
            }
            Reader.MoveToElement();
            if (Reader.IsEmptyElement) {
                Reader.Skip();
                return o;
            }
            Reader.ReadStartElement();
            Reader.MoveToContent();
            int whileIterations12 = 0;
            int readerCount12 = ReaderCount;
            while (Reader.NodeType != System.Xml.XmlNodeType.EndElement && Reader.NodeType != System.Xml.XmlNodeType.None) {
                if (Reader.NodeType == System.Xml.XmlNodeType.Element) {
                    UnknownNode((object)o, @"");
                }
                else {
                    UnknownNode((object)o, @"");
                }
                Reader.MoveToContent();
                CheckReaderCount(ref whileIterations12, ref readerCount12);
            }
            ReadEndElement();
            return o;
        }

        protected override void InitCallbacks() {
        }

        string id11_artist;
        string id9_CTDBResponseMetaTrack;
        string id4_Item;
        string id45_status;
        string id5_CTDBResponseMeta;
        string id37_npar;
        string id34_release;
        string id49_entry;
        string id30_barcode;
        string id22_id;
        string id16_uri;
        string id8_CTDBResponseMetaRelease;
        string id42_trackcrcs;
        string id15_catno;
        string id41_syndrome;
        string id38_stride;
        string id32_track;
        string id24_year;
        string id35_crc32;
        string id14_country;
        string id40_parity;
        string id29_infourl;
        string id18_height;
        string id7_CTDBResponseMetaLabel;
        string id6_CTDBResponseMetaImage;
        string id28_discname;
        string id19_width;
        string id20_primary;
        string id13_date;
        string id1_ctdb;
        string id27_disccount;
        string id2_Item;
        string id3_CTDBResponseEntry;
        string id33_label;
        string id21_source;
        string id17_uri150;
        string id48_message;
        string id39_hasparity;
        string id25_genre;
        string id46_updateurl;
        string id44_CTDBResponse;
        string id50_metadata;
        string id12_extra;
        string id26_discnumber;
        string id47_updatemsg;
        string id23_album;
        string id31_coverart;
        string id36_confidence;
        string id43_toc;
        string id10_name;

        protected override void InitIDs() {
            id11_artist = Reader.NameTable.Add(@"artist");
            id9_CTDBResponseMetaTrack = Reader.NameTable.Add(@"CTDBResponseMetaTrack");
            id4_Item = Reader.NameTable.Add(@"");
            id45_status = Reader.NameTable.Add(@"status");
            id5_CTDBResponseMeta = Reader.NameTable.Add(@"CTDBResponseMeta");
            id37_npar = Reader.NameTable.Add(@"npar");
            id34_release = Reader.NameTable.Add(@"release");
            id49_entry = Reader.NameTable.Add(@"entry");
            id30_barcode = Reader.NameTable.Add(@"barcode");
            id22_id = Reader.NameTable.Add(@"id");
            id16_uri = Reader.NameTable.Add(@"uri");
            id8_CTDBResponseMetaRelease = Reader.NameTable.Add(@"CTDBResponseMetaRelease");
            id42_trackcrcs = Reader.NameTable.Add(@"trackcrcs");
            id15_catno = Reader.NameTable.Add(@"catno");
            id41_syndrome = Reader.NameTable.Add(@"syndrome");
            id38_stride = Reader.NameTable.Add(@"stride");
            id32_track = Reader.NameTable.Add(@"track");
            id24_year = Reader.NameTable.Add(@"year");
            id35_crc32 = Reader.NameTable.Add(@"crc32");
            id14_country = Reader.NameTable.Add(@"country");
            id40_parity = Reader.NameTable.Add(@"parity");
            id29_infourl = Reader.NameTable.Add(@"infourl");
            id18_height = Reader.NameTable.Add(@"height");
            id7_CTDBResponseMetaLabel = Reader.NameTable.Add(@"CTDBResponseMetaLabel");
            id6_CTDBResponseMetaImage = Reader.NameTable.Add(@"CTDBResponseMetaImage");
            id28_discname = Reader.NameTable.Add(@"discname");
            id19_width = Reader.NameTable.Add(@"width");
            id20_primary = Reader.NameTable.Add(@"primary");
            id13_date = Reader.NameTable.Add(@"date");
            id1_ctdb = Reader.NameTable.Add(@"ctdb");
            id27_disccount = Reader.NameTable.Add(@"disccount");
            id2_Item = Reader.NameTable.Add(@"http://db.cuetools.net/ns/mmd-1.0#");
            id3_CTDBResponseEntry = Reader.NameTable.Add(@"CTDBResponseEntry");
            id33_label = Reader.NameTable.Add(@"label");
            id21_source = Reader.NameTable.Add(@"source");
            id17_uri150 = Reader.NameTable.Add(@"uri150");
            id48_message = Reader.NameTable.Add(@"message");
            id39_hasparity = Reader.NameTable.Add(@"hasparity");
            id25_genre = Reader.NameTable.Add(@"genre");
            id46_updateurl = Reader.NameTable.Add(@"updateurl");
            id44_CTDBResponse = Reader.NameTable.Add(@"CTDBResponse");
            id50_metadata = Reader.NameTable.Add(@"metadata");
            id12_extra = Reader.NameTable.Add(@"extra");
            id26_discnumber = Reader.NameTable.Add(@"discnumber");
            id47_updatemsg = Reader.NameTable.Add(@"updatemsg");
            id23_album = Reader.NameTable.Add(@"album");
            id31_coverart = Reader.NameTable.Add(@"coverart");
            id36_confidence = Reader.NameTable.Add(@"confidence");
            id43_toc = Reader.NameTable.Add(@"toc");
            id10_name = Reader.NameTable.Add(@"name");
        }
    }

    public abstract class XmlSerializer1 : System.Xml.Serialization.XmlSerializer {
        protected override System.Xml.Serialization.XmlSerializationReader CreateReader() {
            return new XmlSerializationReader1();
        }
        protected override System.Xml.Serialization.XmlSerializationWriter CreateWriter() {
            return new XmlSerializationWriter1();
        }
    }

    public sealed class CTDBResponseSerializer : XmlSerializer1 {

        public override System.Boolean CanDeserialize(System.Xml.XmlReader xmlReader) {
            return xmlReader.IsStartElement(@"ctdb", @"http://db.cuetools.net/ns/mmd-1.0#");
        }

        protected override void Serialize(object objectToSerialize, System.Xml.Serialization.XmlSerializationWriter writer) {
            ((XmlSerializationWriter1)writer).Write15_ctdb(objectToSerialize);
        }

        protected override object Deserialize(System.Xml.Serialization.XmlSerializationReader reader) {
            return ((XmlSerializationReader1)reader).Read15_ctdb();
        }
    }

    public sealed class CTDBResponseEntrySerializer : XmlSerializer1 {

        public override System.Boolean CanDeserialize(System.Xml.XmlReader xmlReader) {
            return xmlReader.IsStartElement(@"CTDBResponseEntry", @"");
        }

        protected override void Serialize(object objectToSerialize, System.Xml.Serialization.XmlSerializationWriter writer) {
            ((XmlSerializationWriter1)writer).Write16_CTDBResponseEntry(objectToSerialize);
        }

        protected override object Deserialize(System.Xml.Serialization.XmlSerializationReader reader) {
            return ((XmlSerializationReader1)reader).Read16_CTDBResponseEntry();
        }
    }

    public sealed class CTDBResponseMetaSerializer : XmlSerializer1 {

        public override System.Boolean CanDeserialize(System.Xml.XmlReader xmlReader) {
            return xmlReader.IsStartElement(@"CTDBResponseMeta", @"");
        }

        protected override void Serialize(object objectToSerialize, System.Xml.Serialization.XmlSerializationWriter writer) {
            ((XmlSerializationWriter1)writer).Write17_CTDBResponseMeta(objectToSerialize);
        }

        protected override object Deserialize(System.Xml.Serialization.XmlSerializationReader reader) {
            return ((XmlSerializationReader1)reader).Read17_CTDBResponseMeta();
        }
    }

    public sealed class CTDBResponseMetaImageSerializer : XmlSerializer1 {

        public override System.Boolean CanDeserialize(System.Xml.XmlReader xmlReader) {
            return xmlReader.IsStartElement(@"CTDBResponseMetaImage", @"");
        }

        protected override void Serialize(object objectToSerialize, System.Xml.Serialization.XmlSerializationWriter writer) {
            ((XmlSerializationWriter1)writer).Write18_CTDBResponseMetaImage(objectToSerialize);
        }

        protected override object Deserialize(System.Xml.Serialization.XmlSerializationReader reader) {
            return ((XmlSerializationReader1)reader).Read18_CTDBResponseMetaImage();
        }
    }

    public sealed class CTDBResponseMetaLabelSerializer : XmlSerializer1 {

        public override System.Boolean CanDeserialize(System.Xml.XmlReader xmlReader) {
            return xmlReader.IsStartElement(@"CTDBResponseMetaLabel", @"");
        }

        protected override void Serialize(object objectToSerialize, System.Xml.Serialization.XmlSerializationWriter writer) {
            ((XmlSerializationWriter1)writer).Write19_CTDBResponseMetaLabel(objectToSerialize);
        }

        protected override object Deserialize(System.Xml.Serialization.XmlSerializationReader reader) {
            return ((XmlSerializationReader1)reader).Read19_CTDBResponseMetaLabel();
        }
    }

    public sealed class CTDBResponseMetaReleaseSerializer : XmlSerializer1 {

        public override System.Boolean CanDeserialize(System.Xml.XmlReader xmlReader) {
            return xmlReader.IsStartElement(@"CTDBResponseMetaRelease", @"");
        }

        protected override void Serialize(object objectToSerialize, System.Xml.Serialization.XmlSerializationWriter writer) {
            ((XmlSerializationWriter1)writer).Write20_CTDBResponseMetaRelease(objectToSerialize);
        }

        protected override object Deserialize(System.Xml.Serialization.XmlSerializationReader reader) {
            return ((XmlSerializationReader1)reader).Read20_CTDBResponseMetaRelease();
        }
    }

    public sealed class CTDBResponseMetaTrackSerializer : XmlSerializer1 {

        public override System.Boolean CanDeserialize(System.Xml.XmlReader xmlReader) {
            return xmlReader.IsStartElement(@"CTDBResponseMetaTrack", @"");
        }

        protected override void Serialize(object objectToSerialize, System.Xml.Serialization.XmlSerializationWriter writer) {
            ((XmlSerializationWriter1)writer).Write21_CTDBResponseMetaTrack(objectToSerialize);
        }

        protected override object Deserialize(System.Xml.Serialization.XmlSerializationReader reader) {
            return ((XmlSerializationReader1)reader).Read21_CTDBResponseMetaTrack();
        }
    }

    public class XmlSerializerContract : global::System.Xml.Serialization.XmlSerializerImplementation {
        public override global::System.Xml.Serialization.XmlSerializationReader Reader { get { return new XmlSerializationReader1(); } }
        public override global::System.Xml.Serialization.XmlSerializationWriter Writer { get { return new XmlSerializationWriter1(); } }
        System.Collections.Hashtable readMethods = null;
        public override System.Collections.Hashtable ReadMethods {
            get {
                if (readMethods == null) {
                    System.Collections.Hashtable _tmp = new System.Collections.Hashtable();
                    _tmp[@"CUETools.CTDB.CTDBResponse:http://db.cuetools.net/ns/mmd-1.0#:ctdb:True:"] = @"Read15_ctdb";
                    _tmp[@"CUETools.CTDB.CTDBResponseEntry::"] = @"Read16_CTDBResponseEntry";
                    _tmp[@"CUETools.CTDB.CTDBResponseMeta::"] = @"Read17_CTDBResponseMeta";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaImage::"] = @"Read18_CTDBResponseMetaImage";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaLabel::"] = @"Read19_CTDBResponseMetaLabel";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaRelease::"] = @"Read20_CTDBResponseMetaRelease";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaTrack::"] = @"Read21_CTDBResponseMetaTrack";
                    if (readMethods == null) readMethods = _tmp;
                }
                return readMethods;
            }
        }
        System.Collections.Hashtable writeMethods = null;
        public override System.Collections.Hashtable WriteMethods {
            get {
                if (writeMethods == null) {
                    System.Collections.Hashtable _tmp = new System.Collections.Hashtable();
                    _tmp[@"CUETools.CTDB.CTDBResponse:http://db.cuetools.net/ns/mmd-1.0#:ctdb:True:"] = @"Write15_ctdb";
                    _tmp[@"CUETools.CTDB.CTDBResponseEntry::"] = @"Write16_CTDBResponseEntry";
                    _tmp[@"CUETools.CTDB.CTDBResponseMeta::"] = @"Write17_CTDBResponseMeta";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaImage::"] = @"Write18_CTDBResponseMetaImage";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaLabel::"] = @"Write19_CTDBResponseMetaLabel";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaRelease::"] = @"Write20_CTDBResponseMetaRelease";
                    _tmp[@"CUETools.CTDB.CTDBResponseMetaTrack::"] = @"Write21_CTDBResponseMetaTrack";
                    if (writeMethods == null) writeMethods = _tmp;
                }
                return writeMethods;
            }
        }
        System.Collections.Hashtable typedSerializers = null;
        public override System.Collections.Hashtable TypedSerializers {
            get {
                if (typedSerializers == null) {
                    System.Collections.Hashtable _tmp = new System.Collections.Hashtable();
                    _tmp.Add(@"CUETools.CTDB.CTDBResponseEntry::", new CTDBResponseEntrySerializer());
                    _tmp.Add(@"CUETools.CTDB.CTDBResponseMeta::", new CTDBResponseMetaSerializer());
                    _tmp.Add(@"CUETools.CTDB.CTDBResponse:http://db.cuetools.net/ns/mmd-1.0#:ctdb:True:", new CTDBResponseSerializer());
                    _tmp.Add(@"CUETools.CTDB.CTDBResponseMetaTrack::", new CTDBResponseMetaTrackSerializer());
                    _tmp.Add(@"CUETools.CTDB.CTDBResponseMetaRelease::", new CTDBResponseMetaReleaseSerializer());
                    _tmp.Add(@"CUETools.CTDB.CTDBResponseMetaLabel::", new CTDBResponseMetaLabelSerializer());
                    _tmp.Add(@"CUETools.CTDB.CTDBResponseMetaImage::", new CTDBResponseMetaImageSerializer());
                    if (typedSerializers == null) typedSerializers = _tmp;
                }
                return typedSerializers;
            }
        }
        public override System.Boolean CanSerialize(System.Type type) {
            if (type == typeof(global::CUETools.CTDB.CTDBResponse)) return true;
            if (type == typeof(global::CUETools.CTDB.CTDBResponseEntry)) return true;
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMeta)) return true;
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaImage)) return true;
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaLabel)) return true;
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaRelease)) return true;
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaTrack)) return true;
            return false;
        }
        public override System.Xml.Serialization.XmlSerializer GetSerializer(System.Type type) {
            if (type == typeof(global::CUETools.CTDB.CTDBResponse)) return new CTDBResponseSerializer();
            if (type == typeof(global::CUETools.CTDB.CTDBResponseEntry)) return new CTDBResponseEntrySerializer();
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMeta)) return new CTDBResponseMetaSerializer();
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaImage)) return new CTDBResponseMetaImageSerializer();
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaLabel)) return new CTDBResponseMetaLabelSerializer();
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaRelease)) return new CTDBResponseMetaReleaseSerializer();
            if (type == typeof(global::CUETools.CTDB.CTDBResponseMetaTrack)) return new CTDBResponseMetaTrackSerializer();
            return null;
        }
    }
}
