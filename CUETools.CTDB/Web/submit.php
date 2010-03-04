<?php

require_once( 'phpctdb/ctdb.php' );

/**
 * Convert php.ini shorthands to byte
 *
 * @author <gilthans dot NO dot SPAM at gmail dot com>
 * @link   http://de3.php.net/manual/en/ini.core.php#79564
 */
function php_to_byte($v){
    $l = substr($v, -1);
    $ret = substr($v, 0, -1);
    switch(strtoupper($l)){
        case 'P':
            $ret *= 1024;
        case 'T':
            $ret *= 1024;
        case 'G':
            $ret *= 1024;
        case 'M':
            $ret *= 1024;
        case 'K':
            $ret *= 1024;
        break;
    }
    return $ret;
}

// Return the human readable size of a file
// @param int $size a file size
// @param int $dec a number of decimal places

function filesize_h($size, $dec = 1)
{
    $sizes = array('byte(s)', 'kb', 'mb', 'gb');
    $count = count($sizes);
    $i = 0;

    while ($size >= 1024 && ($i < $count - 1)) {
        $size /= 1024;
        $i++;
    }

    return round($size, $dec) . ' ' . $sizes[$i];
}

$file = $_FILES['uploadedfile'];

//echo $file['name'], ini_get('upload_max_filesize');

    // give info on PHP catched upload errors
    if($file['error']) switch($file['error']){
        case 1:
        case 2:
            echo sprintf($lang['uploadsize'],
                filesize_h(php_to_byte(ini_get('upload_max_filesize'))));
            echo "Error ", $file['error'];
            return;
        default:
            echo $lang['uploadfail'];
            echo "Error ", $file['error'];
    }

//if ($_SERVER['HTTP_USER_AGENT'] != "CUETools 205") {
//  echo "user agent ", $_SERVER['HTTP_USER_AGENT'], " is not allowed";
//  return;
//}

$tmpname = $file['tmp_name'];
$size = (@file_exists($tmpname)) ? filesize($tmpname) : 0;
if ($size == 0) {
  echo "no file uploaded";
  return;
}

$dbconn = pg_connect("dbname=ctdb user=ctdb_user")
    or die('Could not connect: ' . pg_last_error());

$id = $_POST['id'];
$target_path = phpCTDB::discid2path($id);
@mkdir($target_path, 0777, true);
$target_file = sprintf("%s/ctdb.bin", $target_path);

$ctdb = new phpCTDB($tmpname);
$merging = @file_exists($target_file);
$ctdb->ParseTOC();
$record = $ctdb->ctdb2pg($id);
unset($ctdb);

$destname = sprintf("%s/%08x.bin", $target_path, $record['ctdbid']);
if(!move_uploaded_file($tmpname, $destname))
  die('error uploading file ' . $tmpname . ' to ' . $destname);

$subres = pg_insert($dbconn, 'submissions', $record);

phpCTDB::pg2ctdb($dbconn, $id);

if ($merging)
  printf("%s has been updated", $id);
else
  printf("%s has been uploaded", $id);
?>
