<?php
include 'table_start.php';
?>
<table class=classy_table cellpadding=3 cellspacing=0><tr bgcolor=#D0D0D0><th>Artist</th><th>Album</th><th>Disc Id</th><th>CTDB Id</th><th>AR</th></tr>
<?php
for($i = $start + $count - 1; $i >= $start; $i--) {
  $record = pg_fetch_array($result, $i); 
  printf('<tr><td class=td_artist><a href=?artist=%s>%s</a></td><td class=td_album>%s</td><td class=td_discid><a href="?discid=%s">%s</a></td><td class=td_ctdbid><a href=show.php?discid=%s&ctdbid=%08x>%08x</a></td><td class=td_ar>%d</td></tr>', urlencode($record['artist']), $record['artist'], $record['title'], $record['discid'], $record['discid'], $record['discid'], $record['ctdbid'], $record['ctdbid'], $record['confidence']);
}
if ($start > 0) printf('<tr><td colspan=5 align=right><a class=style1 href="?start=%d%s">More</a></td></tr>', $count * floor(($start - 1) / $count), $url);
printf("</table>");
include 'table_end.php' ;
?>
