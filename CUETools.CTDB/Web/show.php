<?php
include 'logo_start.php';
require_once( 'phpmb/mbQuery.php' );
require_once( 'phpctdb/ctdb.php' );

$discid = $_GET['discid'];
$ctdbid = $_GET['ctdbid'];
$path = phpCTDB::ctdbid2path($discid, $ctdbid);
$ctdb = new phpCTDB($path);
$disc = $ctdb->db['discs'][0];
//$id = @$disc['MBID']['value'];
$ctdb->ParseTOC();
$artist = @$disc['ART ']['value'];
$title = @$disc['nam ']['value'];
$link = '<a href="http://musicbrainz.org/bare/cdlookup.html?id=' . $ctdb->mbid . '">';
$q = new MusicBrainzQuery();
$rf = new ReleaseFilter();
$mbrr = false;
try {
  $rresults = $q->getReleases( $rf->discId($ctdb->mbid) );
  foreach ( $rresults as  $key => $rr  ) {
    $mbrr = $rr->getRelease();
  }
}
catch ( ResponseError $e ) {
	// echo $e->getMessage() . " ";
}
printf('<center>');
if ($mbrr && $mbrr->getAsin())
{
	include 'table_start.php';
	$imgurl = 'http://ec1.images-amazon.com/images/P/' . $mbrr->getAsin() . '.01.MZZZZZZZ.jpg';
	#$imgurl = 'http://images.amazon.com/images/P/' . $mbrr->getAsin() . '.01._SCLZZZZZZZ_PU_PU-5_.jpg';
	printf('<img src="%s">', $imgurl);
	include 'table_end.php';
	printf('<br><br>');
}
include 'table_start.php';
?>
<table border=0 cellspacing=0 cellpadding=6>
<?php
printf('<tr><td>Full TOC</td><td>%s</td></tr>', $ctdb->fulltoc);
printf('<tr><td>Artist</td><td>%s</td></tr>', $artist);
if ($mbrr && $mbrr->getArtist()->getName() != $artist)
printf('<tr><td>Artist (MB)</td><td>%s</td></tr>', $mbrr->getArtist()->getName());
printf('<tr><td>Title</td><td>%s</td></tr>', $title);
if ($mbrr && $mbrr->getTitle() != $title)
printf('<tr><td>Title (MB)</td><td>%s</td></tr>', $mbrr->getTitle());
printf('<tr><td>MusicbrainzId</td><td>%s%s</a></tr>', $link, $ctdb->mbid);
?>
</table>
<?php include 'table_end.php'; ?>
</center>
</body>
</html>
