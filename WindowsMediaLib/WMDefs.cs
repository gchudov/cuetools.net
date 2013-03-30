#region license

/*
WindowsMediaLib - Provide access to Windows Media interfaces via .NET
Copyright (C) 2008
http://sourceforge.net/projects/windowsmedianet

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#endregion

// The definitions in this namespace duplicate ones found in DirectShowLib.
// To avoid ambiguous references when using both libs, do NOT use
// "using WindowsMediaLib.Defs;"

using System;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Reflection;

namespace WindowsMediaLib.Defs
{
    public static class MediaType
    {
        public static readonly Guid Null = Guid.Empty;

        /// <summary> MEDIATYPE_Video 'vids' </summary>
        public static readonly Guid Video = new Guid(0x73646976, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_Interleaved 'iavs' </summary>
        public static readonly Guid Interleaved = new Guid(0x73766169, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_Audio 'auds' </summary>
        public static readonly Guid Audio = new Guid(0x73647561, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_Text 'txts' </summary>
        public static readonly Guid Texts = new Guid(0x73747874, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_Stream </summary>
        public static readonly Guid Stream = new Guid(0xe436eb83, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIATYPE_VBI </summary>
        public static readonly Guid VBI = new Guid(0xf72a76e1, 0xeb0a, 0x11d0, 0xac, 0xe4, 0x00, 0x00, 0xc0, 0xcc, 0x16, 0xba);

        /// <summary> MEDIATYPE_Midi </summary>
        public static readonly Guid Midi = new Guid(0x7364696D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_File </summary>
        public static readonly Guid File = new Guid(0x656c6966, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_ScriptCommand </summary>
        public static readonly Guid ScriptCommand = new Guid(0x73636d64, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_AUXLine21Data </summary>
        public static readonly Guid AuxLine21Data = new Guid(0x670aea80, 0x3a82, 0x11d0, 0xb7, 0x9b, 0x00, 0xaa, 0x00, 0x37, 0x67, 0xa7);

        /// <summary> MEDIATYPE_Timecode </summary>
        public static readonly Guid Timecode = new Guid(0x0482dee3, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIATYPE_LMRT </summary>
        public static readonly Guid LMRT = new Guid(0x74726c6d, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_URL_STREAM </summary>
        public static readonly Guid URLStream = new Guid(0x736c7275, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIATYPE_AnalogVideo </summary>
        public static readonly Guid AnalogVideo = new Guid(0x0482dde1, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIATYPE_AnalogAudio </summary>
        public static readonly Guid AnalogAudio = new Guid(0x0482dee1, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIATYPE_MPEG2_SECTIONS </summary>
        public static readonly Guid Mpeg2Sections = new Guid(0x455f176c, 0x4b06, 0x47ce, 0x9a, 0xef, 0x8c, 0xae, 0xf7, 0x3d, 0xf7, 0xb5);

        /// <summary> MEDIATYPE_DTVCCData </summary>
        public static readonly Guid DTVCCData = new Guid(0xfb77e152, 0x53b2, 0x499c, 0xb4, 0x6b, 0x50, 0x9f, 0xc3, 0x3e, 0xdf, 0xd7);

        /// <summary> MEDIATYPE_MSTVCaption </summary>
        public static readonly Guid MSTVCaption = new Guid(0xB88B8A89, 0xB049, 0x4C80, 0xAD, 0xCF, 0x58, 0x98, 0x98, 0x5E, 0x22, 0xC1);

        /// <summary> WMMEDIATYPE_Image 'imag' </summary>
        public static readonly Guid Image = new Guid(0x34a50fd8, 0x8aa5, 0x4386, 0x81, 0xfe, 0xa0, 0xef, 0xe0, 0x48, 0x8e, 0x31);

        /// <summary> WMMEDIATYPE_FileTransfer 'fxfr' </summary>
        public static readonly Guid FileTransfer = new Guid(0xd9e47579, 0x930e, 0x4427, 0xad, 0xfc, 0xad, 0x80, 0xf2, 0x90, 0xe4, 0x70);

        /// <summary> WMMEDIATYPE_Text 'text' </summary>
        public static readonly Guid Text = new Guid(0x9bba1ea7, 0x5ab2, 0x4829, 0xba, 0x57, 0x9, 0x40, 0x20, 0x9b, 0xcf, 0x3e);
    }

    public static class MediaSubType
    {
        public static readonly Guid Null = Guid.Empty;

        /// <summary> MEDIASUBTYPE_CLPL </summary>
        public static readonly Guid CLPL = new Guid(0x4C504C43, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_YUYV </summary>
        public static readonly Guid YUYV = new Guid(0x56595559, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IYUV </summary>
        public static readonly Guid IYUV = new Guid(0x56555949, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_YVU9 </summary>
        public static readonly Guid YVU9 = new Guid(0x39555659, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_Y411 </summary>
        public static readonly Guid Y411 = new Guid(0x31313459, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_Y41P </summary>
        public static readonly Guid Y41P = new Guid(0x50313459, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_YUY2 </summary>
        public static readonly Guid YUY2 = new Guid(0x32595559, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_YVYU </summary>
        public static readonly Guid YVYU = new Guid(0x55595659, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_UYVY </summary>
        public static readonly Guid UYVY = new Guid(0x59565955, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_Y211 </summary>
        public static readonly Guid Y211 = new Guid(0x31313259, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_CLJR </summary>
        public static readonly Guid CLJR = new Guid(0x524a4c43, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IF09 </summary>
        public static readonly Guid IF09 = new Guid(0x39304649, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_CPLA </summary>
        public static readonly Guid CPLA = new Guid(0x414c5043, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_MJPG </summary>
        public static readonly Guid MJPG = new Guid(0x47504A4D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_TVMJ </summary>
        public static readonly Guid TVMJ = new Guid(0x4A4D5654, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_WAKE </summary>
        public static readonly Guid WAKE = new Guid(0x454B4157, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_CFCC </summary>
        public static readonly Guid CFCC = new Guid(0x43434643, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IJPG </summary>
        public static readonly Guid IJPG = new Guid(0x47504A49, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_Plum </summary>
        public static readonly Guid PLUM = new Guid(0x6D756C50, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_DVCS </summary>
        public static readonly Guid DVCS = new Guid(0x53435644, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_DVSD </summary>
        public static readonly Guid DVSD = new Guid(0x44535644, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_MDVF </summary>
        public static readonly Guid MDVF = new Guid(0x4656444D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_RGB1 </summary>
        public static readonly Guid RGB1 = new Guid(0xe436eb78, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_RGB4 </summary>
        public static readonly Guid RGB4 = new Guid(0xe436eb79, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_RGB8 </summary>
        public static readonly Guid RGB8 = new Guid(0xe436eb7a, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_RGB565 </summary>
        public static readonly Guid RGB565 = new Guid(0xe436eb7b, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_RGB555 </summary>
        public static readonly Guid RGB555 = new Guid(0xe436eb7c, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_RGB24 </summary>
        public static readonly Guid RGB24 = new Guid(0xe436eb7d, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_RGB32 </summary>
        public static readonly Guid RGB32 = new Guid(0xe436eb7e, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_ARGB1555 </summary>
        public static readonly Guid ARGB1555 = new Guid(0x297c55af, 0xe209, 0x4cb3, 0xb7, 0x57, 0xc7, 0x6d, 0x6b, 0x9c, 0x88, 0xa8);

        /// <summary> MEDIASUBTYPE_ARGB4444 </summary>
        public static readonly Guid ARGB4444 = new Guid(0x6e6415e6, 0x5c24, 0x425f, 0x93, 0xcd, 0x80, 0x10, 0x2b, 0x3d, 0x1c, 0xca);

        /// <summary> MEDIASUBTYPE_ARGB32 </summary>
        public static readonly Guid ARGB32 = new Guid(0x773c9ac0, 0x3274, 0x11d0, 0xb7, 0x24, 0x00, 0xaa, 0x00, 0x6c, 0x1a, 0x01);

        /// <summary> MEDIASUBTYPE_A2R10G10B10 </summary>
        public static readonly Guid A2R10G10B10 = new Guid(0x2f8bb76d, 0xb644, 0x4550, 0xac, 0xf3, 0xd3, 0x0c, 0xaa, 0x65, 0xd5, 0xc5);

        /// <summary> MEDIASUBTYPE_A2B10G10R10 </summary>
        public static readonly Guid A2B10G10R10 = new Guid(0x576f7893, 0xbdf6, 0x48c4, 0x87, 0x5f, 0xae, 0x7b, 0x81, 0x83, 0x45, 0x67);

        /// <summary> MEDIASUBTYPE_AYUV </summary>
        public static readonly Guid AYUV = new Guid(0x56555941, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_AI44 </summary>
        public static readonly Guid AI44 = new Guid(0x34344941, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IA44 </summary>
        public static readonly Guid IA44 = new Guid(0x34344149, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_RGB32_D3D_DX7_RT </summary>
        public static readonly Guid RGB32_D3D_DX7_RT = new Guid(0x32335237, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_RGB16_D3D_DX7_RT </summary>
        public static readonly Guid RGB16_D3D_DX7_RT = new Guid(0x36315237, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_ARGB32_D3D_DX7_RT </summary>
        public static readonly Guid ARGB32_D3D_DX7_RT = new Guid(0x38384137, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_ARGB4444_D3D_DX7_RT </summary>
        public static readonly Guid ARGB4444_D3D_DX7_RT = new Guid(0x34344137, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_ARGB1555_D3D_DX7_RT </summary>
        public static readonly Guid ARGB1555_D3D_DX7_RT = new Guid(0x35314137, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_RGB32_D3D_DX9_RT </summary>
        public static readonly Guid RGB32_D3D_DX9_RT = new Guid(0x32335239, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_RGB16_D3D_DX9_RT </summary>
        public static readonly Guid RGB16_D3D_DX9_RT = new Guid(0x36315239, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_ARGB32_D3D_DX9_RT </summary>
        public static readonly Guid ARGB32_D3D_DX9_RT = new Guid(0x38384139, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_ARGB4444_D3D_DX9_RT </summary>
        public static readonly Guid ARGB4444_D3D_DX9_RT = new Guid(0x34344139, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_ARGB1555_D3D_DX9_RT </summary>
        public static readonly Guid ARGB1555_D3D_DX9_RT = new Guid(0x35314139, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_YV12 </summary>
        public static readonly Guid YV12 = new Guid(0x32315659, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_NV12 </summary>
        public static readonly Guid NV12 = new Guid(0x3231564E, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IMC1 </summary>
        public static readonly Guid IMC1 = new Guid(0x31434D49, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IMC2 </summary>
        public static readonly Guid IMC2 = new Guid(0x32434D49, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IMC3 </summary>
        public static readonly Guid IMC3 = new Guid(0x33434D49, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IMC4 </summary>
        public static readonly Guid IMC4 = new Guid(0x34434D49, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_S340 </summary>
        public static readonly Guid S340 = new Guid(0x30343353, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_S342 </summary>
        public static readonly Guid S342 = new Guid(0x32343353, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_Overlay </summary>
        public static readonly Guid Overlay = new Guid(0xe436eb7f, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_MPEG1Packet </summary>
        public static readonly Guid MPEG1Packet = new Guid(0xe436eb80, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_MPEG1Payload </summary>
        public static readonly Guid MPEG1Payload = new Guid(0xe436eb81, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_MPEG1AudioPayload </summary>
        public static readonly Guid MPEG1AudioPayload = new Guid(0x00000050, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> MEDIATYPE_MPEG1SystemStream </summary>
        public static readonly Guid MPEG1SystemStream = new Guid(0xe436eb82, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_MPEG1System </summary>
        public static readonly Guid MPEG1System = new Guid(0xe436eb84, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_MPEG1VideoCD </summary>
        public static readonly Guid MPEG1VideoCD = new Guid(0xe436eb85, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_MPEG1Video </summary>
        public static readonly Guid MPEG1Video = new Guid(0xe436eb86, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_MPEG1Audio </summary>
        public static readonly Guid MPEG1Audio = new Guid(0xe436eb87, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_Avi </summary>
        public static readonly Guid Avi = new Guid(0xe436eb88, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_Asf </summary>
        public static readonly Guid Asf = new Guid(0x3db80f90, 0x9412, 0x11d1, 0xad, 0xed, 0x00, 0x00, 0xf8, 0x75, 0x4b, 0x99);

        /// <summary> MEDIASUBTYPE_QTMovie </summary>
        public static readonly Guid QTMovie = new Guid(0xe436eb89, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_QTRpza </summary>
        public static readonly Guid QTRpza = new Guid(0x617a7072, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_QTSmc </summary>
        public static readonly Guid QTSmc = new Guid(0x20636d73, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_QTRle </summary>
        public static readonly Guid QTRle = new Guid(0x20656c72, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_QTJpeg </summary>
        public static readonly Guid QTJpeg = new Guid(0x6765706a, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_PCMAudio_Obsolete </summary>
        public static readonly Guid PCMAudio_Obsolete = new Guid(0xe436eb8a, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_PCM </summary>
        public static readonly Guid PCM = new Guid(0x00000001, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> MEDIASUBTYPE_WAVE </summary>
        public static readonly Guid WAVE = new Guid(0xe436eb8b, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_AU </summary>
        public static readonly Guid AU = new Guid(0xe436eb8c, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_AIFF </summary>
        public static readonly Guid AIFF = new Guid(0xe436eb8d, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_dvhd </summary>
        public static readonly Guid dvhd = new Guid(0x64687664, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_dvsl </summary>
        public static readonly Guid dvsl = new Guid(0x6c737664, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_dv25 </summary>
        public static readonly Guid dv25 = new Guid(0x35327664, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_dv50 </summary>
        public static readonly Guid dv50 = new Guid(0x30357664, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_dvh1 </summary>
        public static readonly Guid dvh1 = new Guid(0x31687664, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_Line21_BytePair </summary>
        public static readonly Guid Line21_BytePair = new Guid(0x6e8d4a22, 0x310c, 0x11d0, 0xb7, 0x9a, 0x00, 0xaa, 0x00, 0x37, 0x67, 0xa7);

        /// <summary> MEDIASUBTYPE_Line21_GOPPacket </summary>
        public static readonly Guid Line21_GOPPacket = new Guid(0x6e8d4a23, 0x310c, 0x11d0, 0xb7, 0x9a, 0x00, 0xaa, 0x00, 0x37, 0x67, 0xa7);

        /// <summary> MEDIASUBTYPE_Line21_VBIRawData </summary>
        public static readonly Guid Line21_VBIRawData = new Guid(0x6e8d4a24, 0x310c, 0x11d0, 0xb7, 0x9a, 0x00, 0xaa, 0x00, 0x37, 0x67, 0xa7);

        /// <summary> MEDIASUBTYPE_TELETEXT </summary>
        public static readonly Guid TELETEXT = new Guid(0xf72a76e3, 0xeb0a, 0x11d0, 0xac, 0xe4, 0x00, 0x00, 0xc0, 0xcc, 0x16, 0xba);

        /// <summary> MEDIASUBTYPE_WSS </summary>
        public static readonly Guid WSS = new Guid(0x2791D576, 0x8E7A, 0x466F, 0x9E, 0x90, 0x5D, 0x3F, 0x30, 0x83, 0x73, 0x8B);

        /// <summary> MEDIASUBTYPE_VPS </summary>
        public static readonly Guid VPS = new Guid(0xa1b3f620, 0x9792, 0x4d8d, 0x81, 0xa4, 0x86, 0xaf, 0x25, 0x77, 0x20, 0x90);

        /// <summary> MEDIASUBTYPE_DRM_Audio </summary>
        public static readonly Guid DRM_Audio = new Guid(0x00000009, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_IEEE_FLOAT </summary>
        public static readonly Guid IEEE_FLOAT = new Guid(0x00000003, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_DOLBY_AC3_SPDIF </summary>
        public static readonly Guid DOLBY_AC3_SPDIF = new Guid(0x00000092, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_RAW_SPORT </summary>
        public static readonly Guid RAW_SPORT = new Guid(0x00000240, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_SPDIF_TAG_241h </summary>
        public static readonly Guid SPDIF_TAG_241h = new Guid(0x00000241, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_DssVideo </summary>
        public static readonly Guid DssVideo = new Guid(0xa0af4f81, 0xe163, 0x11d0, 0xba, 0xd9, 0x00, 0x60, 0x97, 0x44, 0x11, 0x1a);

        /// <summary> MEDIASUBTYPE_DssAudio </summary>
        public static readonly Guid DssAudio = new Guid(0xa0af4f82, 0xe163, 0x11d0, 0xba, 0xd9, 0x00, 0x60, 0x97, 0x44, 0x11, 0x1a);

        /// <summary> MEDIASUBTYPE_VPVideo </summary>
        public static readonly Guid VPVideo = new Guid(0x5a9b6a40, 0x1a22, 0x11d1, 0xba, 0xd9, 0x00, 0x60, 0x97, 0x44, 0x11, 0x1a);

        /// <summary> MEDIASUBTYPE_VPVBI </summary>
        public static readonly Guid VPVBI = new Guid(0x5a9b6a41, 0x1a22, 0x11d1, 0xba, 0xd9, 0x00, 0x60, 0x97, 0x44, 0x11, 0x1a);

        /// <summary> MEDIASUBTYPE_AnalogVideo_NTSC_M </summary>
        public static readonly Guid AnalogVideo_NTSC_M = new Guid(0x0482dde2, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_B </summary>
        public static readonly Guid AnalogVideo_PAL_B = new Guid(0x0482dde5, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_D </summary>
        public static readonly Guid AnalogVideo_PAL_D = new Guid(0x0482dde6, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_G </summary>
        public static readonly Guid AnalogVideo_PAL_G = new Guid(0x0482dde7, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_H </summary>
        public static readonly Guid AnalogVideo_PAL_H = new Guid(0x0482dde8, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_I </summary>
        public static readonly Guid AnalogVideo_PAL_I = new Guid(0x0482dde9, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_M </summary>
        public static readonly Guid AnalogVideo_PAL_M = new Guid(0x0482ddea, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_N </summary>
        public static readonly Guid AnalogVideo_PAL_N = new Guid(0x0482ddeb, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_PAL_N_COMBO </summary>
        public static readonly Guid AnalogVideo_PAL_N_COMBO = new Guid(0x0482ddec, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_SECAM_B </summary>
        public static readonly Guid AnalogVideo_SECAM_B = new Guid(0x0482ddf0, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_SECAM_D </summary>
        public static readonly Guid AnalogVideo_SECAM_D = new Guid(0x0482ddf1, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_SECAM_G </summary>
        public static readonly Guid AnalogVideo_SECAM_G = new Guid(0x0482ddf2, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_SECAM_H </summary>
        public static readonly Guid AnalogVideo_SECAM_H = new Guid(0x0482ddf3, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_SECAM_K </summary>
        public static readonly Guid AnalogVideo_SECAM_K = new Guid(0x0482ddf4, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_SECAM_K1 </summary>
        public static readonly Guid AnalogVideo_SECAM_K1 = new Guid(0x0482ddf5, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_AnalogVideo_SECAM_L </summary>
        public static readonly Guid AnalogVideo_SECAM_L = new Guid(0x0482ddf6, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> MEDIASUBTYPE_I420 </summary>
        public static readonly Guid I420 = new Guid(0x30323449, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> WMMEDIASUBTYPE_VIDEOIMAGE </summary>
        public static readonly Guid VideoImage = new Guid(0x1d4a45f2, 0xe5f6, 0x4b44, 0x83, 0x88, 0xf0, 0xae, 0x5c, 0x0e, 0x0c, 0x37);

        /// <summary> WMMEDIASUBTYPE_MPEG2_VIDEO </summary>
        public static readonly Guid Mpeg2Video = new Guid(0xe06d8026, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> WMMEDIASUBTYPE_WebStream </summary>
        public static readonly Guid WebStream = new Guid(0x776257d4, 0xc627, 0x41cb, 0x8f, 0x81, 0x7a, 0xc7, 0xff, 0x1c, 0x40, 0xcc);

        /// <summary> MEDIASUBTYPE_MPEG2_AUDIO </summary>
        public static readonly Guid Mpeg2Audio = new Guid(0xe06d802b, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> MEDIASUBTYPE_DOLBY_AC3 </summary>
        public static readonly Guid DolbyAC3 = new Guid(0xe06d802c, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> MEDIASUBTYPE_DVB_SI </summary>
        public static readonly Guid DvbSI = new Guid(0xe9dd31a3, 0x221d, 0x4adb, 0x85, 0x32, 0x9a, 0xf3, 0x09, 0xc1, 0xa4, 0x08);

        /// <summary> MEDIASUBTYPE_ATSC_SI </summary>
        public static readonly Guid AtscSI = new Guid(0xb3c7397c, 0xd303, 0x414d, 0xb3, 0x3c, 0x4e, 0xd2, 0xc9, 0xd2, 0x97, 0x33);

        /// <summary> MEDIASUBTYPE_MPEG2DATA </summary>
        public static readonly Guid Mpeg2Data = new Guid(0xc892e55b, 0x252d, 0x42b5, 0xa3, 0x16, 0xd9, 0x97, 0xe7, 0xa5, 0xd9, 0x95);

        /// <summary> MEDIASUBTYPE_MPEG2_PROGRAM </summary>
        public static readonly Guid Mpeg2Program = new Guid(0xe06d8022, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> MEDIASUBTYPE_MPEG2_TRANSPORT </summary>
        public static readonly Guid Mpeg2Transport = new Guid(0xe06d8023, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> MEDIASUBTYPE_MPEG2_TRANSPORT_STRIDE </summary>
        public static readonly Guid Mpeg2TransportStride = new Guid(0x138aa9a4, 0x1ee2, 0x4c5b, 0x98, 0x8e, 0x19, 0xab, 0xfd, 0xbc, 0x8a, 0x11);

        /// <summary> MEDIASUBTYPE_None </summary>
        public static readonly Guid None = new Guid(0xe436eb8e, 0x524f, 0x11ce, 0x9f, 0x53, 0x00, 0x20, 0xaf, 0x0b, 0xa7, 0x70);

        /// <summary> MEDIASUBTYPE_H264 </summary>
        public static readonly Guid H264 = new Guid(0x34363248, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_NV24 </summary>
        public static readonly Guid NV24 = new Guid(0x3432564E, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> MEDIASUBTYPE_708_608Data </summary>
        public static readonly Guid Data708_608 = new Guid(0xaf414bc, 0x4ed2, 0x445e, 0x98, 0x39, 0x8f, 0x9, 0x55, 0x68, 0xab, 0x3c);

        /// <summary> MEDIASUBTYPE_DtvCcData </summary>
        public static readonly Guid DtvCcData = new Guid(0xF52ADDAA, 0x36F0, 0x43F5, 0x95, 0xEA, 0x6D, 0x86, 0x64, 0x84, 0x26, 0x2A);

        /// <summary> MEDIASUBTYPE_P422 </summary>
        public static readonly Guid P422 = new Guid(0x32323450, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71);

        /// <summary> WMMEDIASUBTYPE_MP43 </summary>
        public static readonly Guid MP43 = new Guid(0x3334504D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_MP4S </summary>
        public static readonly Guid MP4S = new Guid(0x5334504D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_M4S2 </summary>
        public static readonly Guid M4S2 = new Guid(0x3253344D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMV1 </summary>
        public static readonly Guid WMV1 = new Guid(0x31564D57, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMV2 </summary>
        public static readonly Guid WMV2 = new Guid(0x32564D57, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_MSS1 </summary>
        public static readonly Guid MSS1 = new Guid(0x3153534D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMAudioV9 </summary>
        public static readonly Guid WMAudioV9 = new Guid(0x00000162, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMAudio_Lossless </summary>
        public static readonly Guid WMAudio_Lossless = new Guid(0x00000163, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_MSS2 </summary>
        public static readonly Guid MSS2 = new Guid(0x3253534D, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMSP1 </summary>
        public static readonly Guid WMSP1 = new Guid(0x0000000A, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMSP2 </summary>
        public static readonly Guid WMSP2 = new Guid(0x0000000B, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMV3 </summary>
        public static readonly Guid WMV3 = new Guid(0x33564D57, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMVP </summary>
        public static readonly Guid WMVP = new Guid(0x50564D57, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WVP2 </summary>
        public static readonly Guid WVP2 = new Guid(0x32505657, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMVA </summary>
        public static readonly Guid WMVA = new Guid(0x41564D57, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WVC1 </summary>
        public static readonly Guid WVC1 = new Guid(0x31435657, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMAudioV8 </summary>
        public static readonly Guid WMAudioV8 = new Guid(0x00000161, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMAudioV7 </summary>
        public static readonly Guid WMAudioV7 = new Guid(0x00000161, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_WMAudioV2 </summary>
        public static readonly Guid WMAudioV2 = new Guid(0x00000161, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_ACELPnet </summary>
        public static readonly Guid ACELPnet = new Guid(0x00000130, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);

        /// <summary> WMMEDIASUBTYPE_MP3 </summary>
        public static readonly Guid MP3 = new Guid(0x00000055, 0x0000, 0x0010, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71);
    }

    public static class FormatType
    {
        public static readonly Guid Null = Guid.Empty;

        /// <summary> FORMAT_None </summary>
        public static readonly Guid None = new Guid(0x0F6417D6, 0xc318, 0x11d0, 0xa4, 0x3f, 0x00, 0xa0, 0xc9, 0x22, 0x31, 0x96);

        /// <summary> FORMAT_VideoInfo </summary>
        public static readonly Guid VideoInfo = new Guid(0x05589f80, 0xc356, 0x11ce, 0xbf, 0x01, 0x00, 0xaa, 0x00, 0x55, 0x59, 0x5a);

        /// <summary> FORMAT_VideoInfo2 </summary>
        public static readonly Guid VideoInfo2 = new Guid(0xf72a76A0, 0xeb0a, 0x11d0, 0xac, 0xe4, 0x00, 0x00, 0xc0, 0xcc, 0x16, 0xba);

        /// <summary> FORMAT_WaveFormatEx </summary>
        public static readonly Guid WaveEx = new Guid(0x05589f81, 0xc356, 0x11ce, 0xbf, 0x01, 0x00, 0xaa, 0x00, 0x55, 0x59, 0x5a);

        /// <summary> FORMAT_MPEGVideo </summary>
        public static readonly Guid MpegVideo = new Guid(0x05589f82, 0xc356, 0x11ce, 0xbf, 0x01, 0x00, 0xaa, 0x00, 0x55, 0x59, 0x5a);

        /// <summary> FORMAT_MPEGStreams </summary>
        public static readonly Guid MpegStreams = new Guid(0x05589f83, 0xc356, 0x11ce, 0xbf, 0x01, 0x00, 0xaa, 0x00, 0x55, 0x59, 0x5a);

        /// <summary> FORMAT_DvInfo </summary>
        public static readonly Guid DvInfo = new Guid(0x05589f84, 0xc356, 0x11ce, 0xbf, 0x01, 0x00, 0xaa, 0x00, 0x55, 0x59, 0x5a);

        /// <summary> FORMAT_AnalogVideo </summary>
        public static readonly Guid AnalogVideo = new Guid(0x0482dde0, 0x7817, 0x11cf, 0x8a, 0x03, 0x00, 0xaa, 0x00, 0x6e, 0xcb, 0x65);

        /// <summary> FORMAT_MPEG2Video </summary>
        public static readonly Guid Mpeg2Video = new Guid(0xe06d80e3, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> FORMAT_DolbyAC3 </summary>
        public static readonly Guid DolbyAC3 = new Guid(0xe06d80e4, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> FORMAT_MPEG2Audio </summary>
        public static readonly Guid Mpeg2Audio = new Guid(0xe06d80e5, 0xdb46, 0x11cf, 0xb4, 0xd1, 0x00, 0x80, 0x5f, 0x6c, 0xbb, 0xea);

        /// <summary> FORMAT_525WSS </summary>
        public static readonly Guid WSS525 = new Guid(0xc7ecf04d, 0x4582, 0x4869, 0x9a, 0xbb, 0xbf, 0xb5, 0x23, 0xb6, 0x2e, 0xdf);

        /// <summary> WMFORMAT_Script </summary>
        public static readonly Guid Script = new Guid(0x5c8510f2, 0xdebe, 0x4ca7, 0xbb, 0xa5, 0xf0, 0x7a, 0x10, 0x4f, 0x8d, 0xff);

        /// <summary> WMFORMAT_WebStream </summary>
        public static readonly Guid WebStream = new Guid(0xda1e6b13, 0x8359, 0x4050, 0xb3, 0x98, 0x38, 0x8e, 0x96, 0x5b, 0xf0, 0x0c);
    }

    /// <summary>
    /// From WM_MEDIA_TYPE - When you are done with an instance of this class,
    /// it should be released with FreeWMMediaType() to avoid leaking
    /// </summary>
    [StructLayout(LayoutKind.Sequential), UnmanagedName("WM_MEDIA_TYPE")]
    public class AMMediaType
    {
        public Guid majorType;
        public Guid subType;
        [MarshalAs(UnmanagedType.Bool)] public bool fixedSizeSamples;
        [MarshalAs(UnmanagedType.Bool)] public bool temporalCompression;
        public int sampleSize;
        public Guid formatType;
        public IntPtr unkPtr; // IUnknown Pointer
        public int formatSize;
        public IntPtr formatPtr; // Pointer to a buff determined by formatType
    }


    /// <summary>
    /// From BITMAPINFOHEADER
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 2), UnmanagedName("BITMAPINFOHEADER")]
    public struct BitmapInfoHeader
    {
        public int Size;
        public int Width;
        public int Height;
        public short Planes;
        public short BitCount;
        public int Compression;
        public int ImageSize;
        public int XPelsPerMeter;
        public int YPelsPerMeter;
        public int ClrUsed;
        public int ClrImportant;
    }


    /// <summary>
    /// From VIDEOINFOHEADER
    /// </summary>
    [StructLayout(LayoutKind.Sequential), UnmanagedName("VIDEOINFOHEADER")]
    public class VideoInfoHeader
    {
        public Rectangle SrcRect;
        public Rectangle TargetRect;
        public int BitRate;
        public int BitErrorRate;
        public long AvgTimePerFrame;
        public BitmapInfoHeader BmiHeader;
    }


    /// <summary>
    /// From VIDEOINFOHEADER2
    /// </summary>
    [StructLayout(LayoutKind.Sequential), UnmanagedName("VIDEOINFOHEADER2")]
    public class VideoInfoHeader2
    {
        public Rectangle SrcRect;
        public Rectangle TargetRect;
        public int BitRate;
        public int BitErrorRate;
        public long AvgTimePerFrame;
        public int InterlaceFlags;
        public int CopyProtectFlags;
        public int PictAspectRatioX;
        public int PictAspectRatioY;
        public int ControlFlags;
        public int Reserved2;
        public BitmapInfoHeader BmiHeader;
    }


    /// <summary>
    /// From WAVEFORMAT
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack=2), UnmanagedName("WAVEFORMAT")]
    public class WaveFormat
    {
        public short wFormatTag;        /* format type */
        public short nChannels;         /* number of channels (i.e. mono, stereo, etc.) */
        public int nSamplesPerSec;    /* sample rate */
        public int nAvgBytesPerSec;   /* for buffer estimation */
        public short nBlockAlign;       /* block size of data */
    }

    /// <summary>
    /// From WAVEFORMATEX
    /// </summary>
    [StructLayout(LayoutKind.Sequential), UnmanagedName("WAVEFORMATEX")]
    public class WaveFormatEx : WaveFormat
    {
        public short wBitsPerSample;
        public short cbSize;
    }

    static public class AMToString
    {
        /// <summary>
        /// Produces a usable string that describes the MediaType object
        /// </summary>
        /// <returns>Concatenation of MajorType + SubType + FormatType + Fixed + Temporal + SampleSize.ToString</returns>
        public static string AMMediaTypeToString(AMMediaType pmt)
        {
            return string.Format("{0} {1} {2} {3} {4} {5}",
                MediaTypeToString(pmt.majorType),
                MediaSubTypeToString(pmt.subType),
                MediaFormatTypeToString(pmt.formatType),
                (pmt.fixedSizeSamples ? "FixedSamples" : "NotFixedSamples"),
                (pmt.temporalCompression ? "temporalCompression" : "NottemporalCompression"),
                pmt.sampleSize.ToString());
        }

        /// <summary>
        /// Converts AMMediaType.MajorType Guid to a readable string
        /// </summary>
        /// <returns>MajorType Guid as a readable string or Guid if unrecognized</returns>
        public static string MediaTypeToString(Guid guid)
        {
            // Walk the MediaSubType class looking for a match
            return WalkClass(typeof(MediaType), guid);
        }

        /// <summary>
        /// Converts the AMMediaType.SubType Guid to a readable string
        /// </summary>
        /// <returns>SubType Guid as a readable string or Guid if unrecognized</returns>
        public static string MediaSubTypeToString(Guid guid)
        {
            // Walk the MediaSubType class looking for a match
            string s = WalkClass(typeof(MediaSubType), guid);

            // There is a special set of Guids that contain the FourCC code
            // as part of the Guid.  If we haven't found a name, check to see if
            // it is one of those.
            if (s == guid.ToString() && FourCC.IsA4ccSubtype(guid))
            {
                s = new FourCC(guid).ToString();
            }

            return s;
        }

        /// <summary>
        /// Converts the AMMediaType.FormatType Guid to a readable string
        /// </summary>
        /// <returns>FormatType Guid as a readable string or Guid if unrecognized</returns>
        public static string MediaFormatTypeToString(Guid guid)
        {
            // Walk the FormatType class looking for a match
            return WalkClass(typeof(FormatType), guid);

        }

        /// <summary>
        /// Use reflection to walk a class looking for a property containing a specified guid
        /// </summary>
        /// <param name="MyType">Class to scan</param>
        /// <param name="guid">Guid to scan for</param>
        /// <returns>String representing property name that matches, or Guid.ToString() for no match</returns>
        private static string WalkClass(Type MyType, Guid guid)
        {
            object o = null;

            // Read the fields from the class
            FieldInfo[] Fields = MyType.GetFields();

            // Walk the returned array
            foreach (FieldInfo m in Fields)
            {
                // Read the value of the property.  The parameter is ignored.
                o = m.GetValue(o);

                // Compare it with the sought value
                if ((Guid)o == guid)
                {
                    return m.Name;
                }
            }

            return guid.ToString();
        }
    }

}
