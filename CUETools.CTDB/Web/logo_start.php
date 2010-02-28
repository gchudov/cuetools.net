<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title>CUETools DB</title>
<style type="text/css">
.style1 {
	font-size: x-small;
	font-family: Arial, Helvetica, sans-serif;
}
.style_fixed {
	#font-size: x-small;
	font-family: Courier;
}
.style_logo {
	font-size: x-large;
	font-family: Arial, Helvetica, sans-serif;
}
.td_status {
	font-size: x-small;
  text-align: center;
	font-family: sans-serif;
}
.td_artist {
  padding: 1px 10px;
}
.td_album {
  padding: 1px 10px;
}
.td_discid {
  padding: 1px 5px;
	font-family: Courier;
}
.td_ctdbid {
  padding: 1px 5px;
	font-family: Courier;
}
.classy_table {
  border-style:none;
#  border-color-right: #D0D0D0
#  border-color-left: #D0D0D0
}
</style>
</head>
<body>
<table border=0 cellspacing=0 cellpadding=3 align=center>
<tr>
<td rowspan=3><img src=ctdb.png></td>
<td colspan=2 class=td_status>
<?php
printf("%d unique discs", filesize("parity/list.txt")/57);
?>
</td>
</tr>
<tr>
<td class=style_logo colspan=2>CUETools Database</td>
</tr>
<tr align=center>
<td>About</td>
<td><a href=http://www.cuetools.net>CUETools</a></td>
</tr>
</table>
<br clear=all>
