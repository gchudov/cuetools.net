<?php
$listfile="parity/list.txt";
$last_modified_time = filemtime($listfile);
$etag = md5_file($listfile);

header("Last-Modified: ".gmdate("D, d M Y H:i:s", $last_modified_time)." GMT");
header("ETag: $etag");

if (@strtotime($_SERVER['HTTP_IF_MODIFIED_SINCE']) == $last_modified_time ||
    @trim($_SERVER['HTTP_IF_NONE_MATCH']) == $etag) {
    header("HTTP/1.1 304 Not Modified");
    exit;
}
include 'logo_start.php'; 
require_once( 'phpmb/mbQuery.php' );
require_once( 'phpctdb/ctdb.php' );

$lines=false;
$totaldiscs=filesize($listfile)/57;
$fp = fopen($listfile,'r');
$count = 10;
$start = @$_GET['start'];
if ($start == "" || $start > $totaldiscs || $start < 0) $start = $totaldiscs - $count;
fseek($fp, 57 * $start, SEEK_SET);
$n = $count;
while($line=fgets($fp)) {
  $name = trim($line);
  if (file_exists($name))
		$lines[]=$name;
  if (--$n <= 0)
		break;
}
fclose($fp);
printf("<center><h3>Recent additions:</h3>");
include 'table_start.php';
?>
      <table class=classy_table cellpadding=3 cellspacing=0><tr bgcolor=#D0D0D0><th>Artist</th><th>Album</th><th>Disc Id</th><th>CTDB Id</th></tr>
<?php
      //<tr background="img/bg_top_border.jpg"><td></td><td width=8 rowspan=12 bgcolor=#FFFFFF></td><td background="img/bg_right_border.jpg" width=4 rowspan=12></td><td></td><td></td><td></td></tr>
$imgs = '';
if ($lines) foreach(array_reverse($lines) as $line) {
	$ctdb = new phpCTDB($line);
  $disc = $ctdb->db['discs'][0];
  $ctdb->ParseToc();
	//$id = @$disc['MBID']['value'];
	$artist = @$disc['ART ']['value'];
	$title = @$disc['nam ']['value'];
  $discid = substr($line,13,30);
  $ctdbid = $ctdb->db['discs'][0]['CRC ']['int'];
  if ($artist == "" && $title == "")
  {
		//$q = new MusicBrainzQuery(new WebService('db4.cuetools.net'));
		$q = new MusicBrainzQuery();
		$rf = new ReleaseFilter();
		try {
			$rresults = $q->getReleases( $rf->discId($ctdb->mbid) );
			foreach ( $rresults as  $key => $rr  ) {
		  	$rr = $rr->getRelease();
				$artist = $rr->getArtist()->getName();
				$title = $rr->getTitle();
        $imgs = $imgs . '<img src="http://ec1.images-amazon.com/images/P/' . $rr->getAsin() . '.01.MZZZZZZZ.jpg">';
			}
		} 
		catch ( ResponseError $e ) {
		  // echo $e->getMessage() . " ";
		}
  }
  printf('<tr><td class=td_artist>%s</td><td class=td_album>%s</td><td class=td_discid><a href="http://musicbrainz.org/bare/cdlookup.html?id=%s">%s</a></td><td class=td_ctdbid><a href=show.php?discid=%s&ctdbid=%08x>%08x</a></td></tr>', htmlspecialchars($artist), htmlspecialchars($title), $ctdb->mbid, $discid, $discid, $ctdbid, $ctdbid);
}
printf('<tr><td colspan=4 align=right><a class=style1 href="?start=%d">More</a></td></tr>', $count * floor(($start - 1) / $count));
printf("</table>");
include 'table_end.php' ;
//printf('%s', $imgs);
printf("</center>");
?>
</body>
</html>
