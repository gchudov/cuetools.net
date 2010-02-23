<?php

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

function BigEndian2Int($byte_word, $signed = false) {

    $int_value = 0;
    $byte_wordlen = strlen($byte_word);

    for ($i = 0; $i < $byte_wordlen; $i++) {
        $int_value += ord($byte_word{$i}) * pow(256, ($byte_wordlen - 1 - $i));
    }

    if ($signed) {
        $sign_mask_bit = 0x80 << (8 * ($byte_wordlen - 1));
        if ($int_value & $sign_mask_bit) {
            $int_value = 0 - ($int_value & ($sign_mask_bit - 1));
        }
    }

    return $int_value;
}

function LittleEndian2String($number, $minbytes=1, $synchsafe=false) {
    $intstring = '';
    while ($number > 0) {
        if ($synchsafe) {
            $intstring = $intstring.chr($number & 127);
            $number >>= 7;
        } else {
            $intstring = $intstring.chr($number & 255);
            $number >>= 8;
        }
    }
    return str_pad($intstring, $minbytes, "\x00", STR_PAD_RIGHT);
}

function BigEndian2String($number, $minbytes=1, $synchsafe=false) {
  return strrev(LittleEndian2String($number, $minbytes, $synchsafe));
}

function unparse_atom($fp, $atom)
{
//  printf('unparse_atom(%s)<br>', $atom['name']);
  $offset = ftell($fp);
  fwrite($fp, BigEndian2String(0, 4));
  fwrite($fp, $atom['name']);
  if ($atom['subatoms'])
    foreach ($atom['subatoms'] as $subatom)
      unparse_atom($fp, $subatom);
  else if ($atom['value'])
    fwrite($fp, $atom['value']);
  else
    die(sprintf("couldn't write long atom %s: size %d", $atom['name'], $atom['size']));
  $pos = ftell($fp);
  fseek($fp, $offset, SEEK_SET);
  fwrite($fp, BigEndian2String($pos - $offset, 4));
  fseek($fp, $pos, SEEK_SET);
}

function parse_container_atom($fp, $offset, $len)
{
//  printf('parse_container_atom(%d, %d)<br>', $offset, $len);
  $atoms = false;
  $fin = $offset + $len;
  while ($offset < $fin) {
    fseek($fp, $offset, SEEK_SET);
    $atom_header = fread($fp, 8);
    $atom_size = BigEndian2Int(substr($atom_header, 0, 4));
    $atom_name = substr($atom_header, 4, 4);
    $atom['name'] = $atom_name;
    $atom['size'] = $atom_size - 8;
    $atom['offset'] = $offset + 8;
    if ($atom_size - 8 <= 32)
      $atom['value'] = fread($fp, $atom_size - 8);
    else
      $atom['value'] = false;
//    echo $offset, ":", $atom_size, ":", $atom_name, '<br>';
    if ($atom_name == 'CTDB' || $atom_name == 'DISC' || $atom_name == 'TOC ')
    {
      $atom['subatoms'] = parse_container_atom($fp, $offset + 8, $atom_size - 8);
      foreach ($atom['subatoms'] as $param)
        switch ($param['name']) {
          case 'HEAD':
          case 'CRC ':
	  case 'NPAR':
	  case 'CONF':
	  case 'PAR ':
            $atom[$param['name']] = $param;
            break;
	  case 'DISC':
	    $atom['discs'][] = $param; 
            break;
        }
    } else
      $atom['subatoms'] = false;
    switch ($atom_name)
    {
      case 'CRC ':
      case 'NPAR':
      case 'CONF':
        $atom['int'] = BigEndian2Int($atom['value']);
        break;
    }
    $offset += $atom_size;
    $atoms[] = $atom;
  }
  if ($offset > $fin)
    die("bad atom");
  return $atoms;
}

function get_chunk_offset($fp, $offset, $maxlen, $names, $namepos, &$res, &$len)
{
//  printf('get_chunk_offset(%d, %d, [%d]%s)<br>', $offset, $maxlen, $namepos, $names[$namepos]);
  $subatoms = parse_container_atom($fp, $offset, $maxlen);
  if (!$subatoms) return -1;
  foreach($subatoms as $atom)
    if ($atom['name'] == $names[$namepos])
    {
      if ($namepos + 1 >= count($names))
      {
        $res = $atom['offset'];
        $len = $atom['size'];
        return 0;
      }
      return get_chunk_offset($fp, $atom['offset'], $atom['size'], $names, $namepos + 1, $res, $len);
    }
  return -1;
}

function chunk_offset($fp, $offset, $maxlen, $path, &$res, &$len)
{
//  printf('chunk_offset(%d, %d, %s)<br>', $offset, $maxlen, $path);
  return get_chunk_offset($fp, $offset, $maxlen, explode(".", $path), 0, $res, $len);
}

function read_chunk($fp, $offset, $maxlen, $path, $len = 32)
{
//  printf('read_chunk(%d, %d, %s)<br>', $offset, $maxlen, $path);
  if (chunk_offset($fp, $offset, $maxlen, $path, $chunk_offset, $chunk_length) < 0) return;
  if ($chunk_length > $len) return;
  fseek($fp, $chunk_offset, SEEK_SET);
  return fread($fp, $chunk_length);
}

function read_int($fp, $offset, $len, $path)
{
//  printf('read_int(%d, %d, %s)<br>', $offset, $len, $path);
  return BigEndian2Int(read_chunk($fp, $offset, $len, $path, 4));
}

function copy_data($srcfp, $srcoffset, $dstfp, $dstoffset, $length)
{
  fseek($srcfp, $srcoffset, SEEK_SET);
  fseek($dstfp, $dstoffset, SEEK_SET);
  fwrite($dstfp, fread($srcfp, $length));
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
$err = sscanf($id, "%03d-%04x%04x-%04x%04x-%04x%04x", $tracks, $id1a, $id1b, $id2a, $id2b, $cddbida, $cddbidb);
$parsedid = sprintf("%03d-%04x%04x-%04x%04x-%04x%04x", $tracks, $id1a, $id1b, $id2a, $id2b, $cddbida, $cddbidb);
if ($id != $parsedid) {
  echo "bad id ", $id;
  return;
}

$fp = fopen($tmpname, 'rb');
if (chunk_offset($fp, 0, $size, 'CTDB.DISC', $disc_offset, $disc_length) < 0)
  die("bad file");
$head = read_chunk($fp, 0, $size, 'CTDB.HEAD', 20);
$crc = read_int($fp, $disc_offset, $disc_length, 'CRC ');
$npar = read_int($fp, $disc_offset, $disc_length, 'NPAR');
$tool = read_chunk($fp, $disc_offset, $disc_length, 'TOOL');
$version = BigEndian2Int(substr($head,0,4));
$disccount = BigEndian2Int(substr($head,4,4));
$total = BigEndian2Int(substr($head,8,4));
fclose($fp);

$target_path = sprintf("parity/%x/%x/%x", $id1b & 15, ($id1b >> 4) & 15, ($id1b >> 8) & 15);
$target_file = sprintf("%s/dBCT-%s.bin", $target_path, $parsedid);

@mkdir($target_path, 0777, true);

if ($npar < 8 || $npar > 16 || $version != 256 || $disccount != 1) {
  printf("bad file: version=%d, disccount=%d, total=%d, npar=%d, tool=%s", 
    $version, $disccount, $total, $npar, $tool);
  return;
}

if (!@file_exists($target_file)) {
  if(!move_uploaded_file($tmpname, $target_file)) {
    echo "there was an error uploading the file, please try again!";
    return;
  }
  printf("%s has been uploaded", $parsedid);
  return;
}

$fp = fopen($tmpname, 'rb');
$fpstats = fstat($fp);
$db = parse_container_atom($fp, 0, $fpstats['size']);
foreach ($db as $entry) if($entry['name'] == 'CTDB') $ctdb = $entry;

if (@file_exists($target_file)) {
  $fp1 = fopen($target_file, 'rb');
  $fp1stats = fstat($fp1);
  $db1 = parse_container_atom($fp1, 0, $fp1stats['size']);
  foreach ($db1 as $entry) if($entry['name'] == 'CTDB') $ctdb1 = $entry;
}

$ftyp['name'] = 'ftyp';
$ftyp['value'] = 'CTDB';

$newctdb['name'] = 'CTDB';
  $newhead['name'] = 'HEAD';
    $newtotal['name'] = 'TOTL';
    $newtotal['value'] = $ctdb1 ? BigEndian2String(BigEndian2Int(substr($ctdb1['HEAD']['value'],8,4)) + 1,4) : substr($ctdb['HEAD']['value'],8,4);
  $newhead['subatoms'][] = $newtotal;
$newctdb['subatoms'][] = $newhead;
$discs = 0;
foreach ($ctdb['subatoms'] as $disc)
  if ($disc['name'] == 'DISC') 
  {
    $crc = $disc['CRC ']['value'];

      $newdisc = false;
      $newdisc['name'] = 'DISC';
      $newdisc['subatoms'][] = $disc['CRC '];
      $newdisc['subatoms'][] = $disc['NPAR'];
      $newdisc['subatoms'][] = $disc['CONF'];
        fseek($fp, $disc['PAR ']['offset']);
        $newpar['name'] = 'PAR ';
        $newpar['value'] = fread($fp, 16);
      $newdisc['subatoms'][] = $newpar;
    $newctdb['subatoms'][] = $newdisc;
    $discs++;
  }
if ($discs > 1)
  die('One disc at a time, please');
if ($discs < 1)
  die('No disc records found');
if ($ctdb1)
foreach ($ctdb1['subatoms'] as $disc)
  if ($disc['name'] == 'DISC') 
  {
    if ($crc == $disc['CRC ']['value'])
      die("duplicate entry");

      $newdisc = false;
      $newdisc['name'] = 'DISC';
      $newdisc['subatoms'][] = $disc['CRC '];
      $newdisc['subatoms'][] = $disc['NPAR'];
      $newdisc['subatoms'][] = $disc['CONF'];
        fseek($fp1, $disc['PAR ']['offset']);
        $newpar['name'] = 'PAR ';
        $newpar['value'] = fread($fp1, 16);
      $newdisc['subatoms'][] = $newpar;
    $newctdb['subatoms'][] = $newdisc;
  }


$destpath = sprintf("%s/%s/", $target_path, $parsedid);
@mkdir($destpath, 0777, true);

$tname = sprintf("%s/ctdb.tmp", $destpath);
$tfp = fopen($tname, 'wb');
unparse_atom($tfp,$ftyp);
unparse_atom($tfp,$newctdb);
fclose($tfp);

$crca = BigEndian2Int(substr($crc,0,2));
$crcb = BigEndian2Int(substr($crc,2,2));
$destname = sprintf("%s/%04x%04x.bin", $destpath, $crca, $crcb);
if(!move_uploaded_file($tmpname, $destname))
  die('error uploading file');

if(!rename($tname,sprintf("%s/ctdb.bin", $destpath)))
  die('error uploading file');

fclose($fp);
if ($fp1) fclose($fp1);

printf("%s has been updated", $parsedid);
?>
