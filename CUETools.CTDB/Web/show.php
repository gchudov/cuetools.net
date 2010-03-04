<?php
//mb_http_output('UTF-8');
//mb_internal_encoding('UTF-8');
include 'logo_start.php';
//require_once( 'phpmb/mbQuery.php' );
require_once( 'phpctdb/ctdb.php' );

$discid = $_GET['discid'];
$ctdbid = $_GET['ctdbid'];
$path = phpCTDB::ctdbid2path($discid, $ctdbid);

$last_modified_time = filemtime($path);
$etag = md5_file($path);

header("Last-Modified: ".gmdate("D, d M Y H:i:s", $last_modified_time)." GMT");
header("ETag: $etag");

if (@strtotime($_SERVER['HTTP_IF_MODIFIED_SINCE']) == $last_modified_time ||
    @trim($_SERVER['HTTP_IF_NONE_MATCH']) == $etag) {
    header("HTTP/1.1 304 Not Modified");
    exit;
}

$result = pg_query_params($dbconn, "SELECT * FROM submissions WHERE discid=$1 AND ctdbid=$2", array($discid,  (int)phpCTDB::Hex2Int($ctdbid)))
  or die('Query failed: ' . pg_last_error());
if (pg_num_rows($result) < 1) die('not found');
if (pg_num_rows($result) > 1) die('not unique');
$record = pg_fetch_array($result);
pg_free_result($result);

$mbid = phpCTDB::toc2mbid($record['fulltoc']);
$mbmeta = phpCTDB::mblookup($mbid);

printf('<center>');
$imgfound = false;
if ($mbmeta)
	foreach ($mbmeta as $mbr)
		if ($mbr['coverarturl'])
		{
			if (!$imgfound) include 'table_start.php';
			printf('<img src="%s">', $mbr['coverarturl']);
			$imgfound = true;
		}
if ($imgfound) {
	include 'table_end.php';
	printf('<br><br>');
}
include 'table_start.php';
?>
<table border=0 cellspacing=0 cellpadding=6>
<?php
printf('<tr><td>Disc ID</td><td>%s</td></tr>', $discid);
printf('<tr><td>CTDB ID</td><td>%s</td></tr>', $ctdbid);
printf('<tr><td>Musicbrainz ID</td><td><a href="http://musicbrainz.org/bare/cdlookup.html?toc=%s">%s</a> (%s)</tr>', phpCTDB::toc2mbtoc($record['fulltoc']), $mbid, $mbmeta ? count($mbmeta) : "-");
//printf('<tr><td>Full TOC</td><td>%s</td></tr>', $record['fulltoc']);
printf('<tr><td colspan=2>');
?>
<table width=100% class=classy_table cellpadding=3 cellspacing=0><tr bgcolor=#D0D0D0><th>Track</th><th>Start</th><th>Length</th><th>Start sector</th><th>End sector</th></tr>
<?php
function TimeToString($time)
{
	$frame = $time % 75;
  $time = floor($time/75);
  $sec = $time % 60;
  $time = floor($time/60);
  $min = $time;
  return sprintf('%d:%02d.%02d',$min,$sec,$frame);
}
$ids = explode(' ', $record['fulltoc']);
for ($tr = 4; $tr < count($ids); $tr++)
{
  $trno = $tr + $ids[0] - 4;
  $trstart = $ids[$tr] - 150;
  $trend = (($tr + 1) < count($ids) ? $ids[$tr+1] : $ids[3]) - 151;
  $trstartmsf = TimeToString($trstart);
  $trlenmsf = TimeToString($trend + 1 - $trstart);
	printf('<tr><td class=td_ar>%d</td><td class=td_ar>%s</td><td class=td_ar>%s</td><td class=td_ar>%d</td><td class=td_ar>%d</td></tr>', $trno, $trstartmsf, $trlenmsf, $trstart, $trend);
}
printf("</table>");
printf('</td></tr>');
if ($record['artist'] != '')
	printf('<tr><td>Artist</td><td>%s</td></tr>', $record['artist']);
if ($mbmeta)
	foreach ($mbmeta as $mbr)
		if ($mbr['artistname'] != $record['artist'])
			printf('<tr><td>Artist (MB)</td><td>%s</td></tr>', $mbr['artistname']);
if ($record['title'] != '')
	printf('<tr><td>Title</td><td>%s</td></tr>', $record['title']);
if ($mbmeta)
	foreach ($mbmeta as $mbr)
		if ($mbr['albumname'] != $record['title'])
			printf('<tr><td>Title (MB)</td><td>%s</td></tr>', $mbr['albumname']);
?>
</table>
<?php include 'table_end.php'; ?>
</center>
</body>
</html>
