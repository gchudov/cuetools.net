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
</style>
</head>
<body>
<H1>CUETools Database</H1>
<?php
require_once( 'phpmb/mbQuery.php' );
require_once( 'phpctdb/ctdb.php' );

$lines=false;
$fp = fopen('parity/list.txt','r');
$count = 10;
fseek($fp, -57 * $count, SEEK_END);
while($line=fgets($fp)) {
  $name = trim($line);
  if (file_exists($name))
		$lines[]=$name;
  if (--$count <= 0)
		break;
}
fclose($fp);
printf("<h3>Recent additions:</h3>");
printf("<table border=3><tr><th>Artist</th><th>Album</th><th>Disc Id</th><th>CTDB Id</th></tr>");
if ($lines) foreach(array_reverse($lines) as $line) {
	$ctdb = new phpCTDB($line);
  $disc = $ctdb->db['discs'][0];
	$id = @$disc['MBID']['value'];
	$artist = @$disc['ART ']['value'];
	$title = @$disc['nam ']['value'];
  $link = $id == "" ? "<a>" : '<a href="http://musicbrainz.org/bare/cdlookup.html?id=' . $id . '">';
  //echo $line . ':' . $id . '<br>';
  if ($artist == "" && $title == "" && $id != "") {
		//$q = new MusicBrainzQuery(new WebService('db4.cuetools.net'));
		$q = new MusicBrainzQuery();
		$rf = new ReleaseFilter();
		try {
			$rresults = $q->getReleases( $rf->discId($id) );
			foreach ( $rresults as  $key => $rr  ) {
		  	$rr = $rr->getRelease();
				$artist = $rr->getArtist()->getName();
				$title = $rr->getTitle();
			}
		} 
		catch ( ResponseError $e ) {
		  // echo $e->getMessage() . " ";
		}
  }
  printf("<tr><td>%s</td><td>%s</td><td class=style_fixed>%s%s</a></td><td class=style_fixed>%x</td></tr>", $artist, $title, $link, substr($line,13,30), $ctdb->db['discs'][0]['CRC ']['int']);
}
printf("</table>");
printf("<h5>Status: %d unique discs.</h5>", filesize("parity/list.txt")/57);
?>
</body>
