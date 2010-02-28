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

$id = $_POST['id'];
$target_path = phpCTDB::discid2path($id);
@mkdir($target_path, 0777, true);
$target_file = sprintf("%s/ctdb.bin", $target_path);

$ctdb = new phpCTDB($tmpname);
$ctdb1 = @file_exists($target_file) ? new phpCTDB($target_file) : false;
$merging = $ctdb1 ? true : false;

$ftyp['name'] = 'ftyp';
$ftyp['value'] = 'CTDB';

$newctdb['name'] = 'CTDB';
  $newhead['name'] = 'HEAD';
    $newtotal['name'] = 'TOTL';
    $newtotal['value'] = $ctdb1 ? phpCTDB::BigEndian2String($ctdb1->db['HEAD']['TOTL']['int'] + 1,4) : $ctdb->db['HEAD']['TOTL']['value'];
  $newhead['subatoms'][] = $newtotal;
$newctdb['subatoms'][] = $newhead;

if ($ctdb1)
foreach ($ctdb1->db['subatoms'] as $disc)
  if ($disc['name'] == 'DISC') 
  {
    if ($crc == $disc['CRC ']['value'])
      die("duplicate entry");

      $newdisc = false;
      $newdisc['name'] = 'DISC';
      $newdisc['subatoms'][] = $disc['CRC '];
      $newdisc['subatoms'][] = $disc['NPAR'];
      $newdisc['subatoms'][] = $disc['CONF'];
        $newpar['name'] = 'PAR ';
        $newpar['value'] = $ctdb1->read($disc['PAR ']['offset'], 16);
      $newdisc['subatoms'][] = $newpar;
    $newctdb['subatoms'][] = $newdisc;
  }

$discs = 0;
foreach ($ctdb->db['subatoms'] as $disc)
  if ($disc['name'] == 'DISC') 
  {
    $crc = $disc['CRC ']['value'];

      $newdisc = false;
      $newdisc['name'] = 'DISC';
      $newdisc['subatoms'][] = $disc['CRC '];
      $newdisc['subatoms'][] = $disc['NPAR'];
      $newdisc['subatoms'][] = $disc['CONF'];
        $newpar['name'] = 'PAR ';
        $newpar['value'] = $ctdb->read($disc['PAR ']['offset'], 16);
      $newdisc['subatoms'][] = $newpar;
    $newctdb['subatoms'][] = $newdisc;
    $discs++;
  }
if ($discs > 1)
  die('One disc at a time, please');
if ($discs < 1)
  die('No disc records found');

$tname = sprintf("%s/ctdb.tmp", $target_path);
$tfp = fopen($tname, 'wb');
phpCTDB::unparse_atom($tfp,$ftyp);
phpCTDB::unparse_atom($tfp,$newctdb);
fclose($tfp);
unset($ctdb);
unset($ctdb1);

$crca = phpCTDB::BigEndian2Int(substr($crc,0,2));
$crcb = phpCTDB::BigEndian2Int(substr($crc,2,2));
$destname = sprintf("%s/%04x%04x.bin", $target_path, $crca, $crcb);
if(!move_uploaded_file($tmpname, $destname))
  die('error uploading file ' . $tmpname . ' to ' . $destname);

if(!rename($tname,sprintf("%s/ctdb.bin", $target_path)))
  die('error uploading file ' . $target_path);

$listfp = fopen("parity/list.txt", 'a');
fwrite($listfp, $destname);
fwrite($listfp, "\n");
fclose($listfp);

if ($merging)
  printf("%s has been updated", $id);
else
  printf("%s has been uploaded", $id);
?>
