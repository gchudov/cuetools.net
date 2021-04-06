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

function get_chunk_offset($fp, $offset, $filelen, $names, $namepos)
{
//  echo $offset, ":", $filelen, ":", $names, ":", $namepos, ":", count($names), "<br>";
  if ($namepos >= count($names))
    return $offset;
  while ($offset < $filelen) {
    fseek($fp, $offset, SEEK_SET);
    $atom_header = fread($fp, 8);
    $atom_size = BigEndian2Int(substr($atom_header, 0, 4));
    $atom_name = substr($atom_header, 4, 4);
//    echo $atom_size, ":", $atom_name, ":", $names[$namepos], '<br>';
    if ($names[$namepos] == $atom_name)
        return get_chunk_offset($fp, $offset + 8, $filelen, $names, $namepos + 1);
    $offset += $atom_size;
  }
  return -1;
}

function chunk_offset($fp, $offset, $filelen, $path)
{
  $names = explode(".", $path);
  return get_chunk_offset($fp, $offset, $filelen, $names, 0);
}

function read_chunk($fp, $offset, $filelen, $path, $len)
{
  $offset = chunk_offset($fp, $offset, $filelen, $path);
  if ($offset < 0) return;
  fseek($fp, $offset, SEEK_SET);
  return fread($fp, $len);
}

function read_int($fp, $offset, $filelen, $path)
{
  return BigEndian2Int(read_chunk($fp, $offset, $filelen, $path, 4));
}

$file = $_FILES['uploadedfile'];

//echo $file['name'], ini_get('upload_max_filesize');

    // give info on PHP caught upload errors
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

$fp = fopen($tmpname, 'rb');
$head = read_chunk($fp, 0, $size, 'CTDB.HEAD', 20);
$npar = read_int($fp, 0, $size, 'CTDB.DISC.NPAR');
$version = BigEndian2Int(substr($head,0,4));
$disccount = BigEndian2Int(substr($head,4,4));
$total = BigEndian2Int(substr($head,8,4));
printf("npar=%d, disccount=%d, total=%d,", $npar, $disccount, $total);
fclose($fp);

$id = $_POST['id'];
$err = sscanf($id, "%03d-%04x%04x-%04x%04x-%04x%04x", $tracks, $id1a, $id1b, $id2a, $id2b, $cddbida, $cddbidb);
$parsedid = sprintf("%03d-%04x%04x-%04x%04x-%04x%04x", $tracks, $id1a, $id1b, $id2a, $id2b, $cddbida, $cddbidb);
if ($id != $parsedid) {
  echo "bad id ", $id;
  return;
}
$target_path = sprintf("parity/%x/%x/%x", $id1b & 15, ($id1b >> 4) & 15, ($id1b >> 8) & 15);
$target_file = sprintf("%s/dBCT-%s.bin", $target_path, $parsedid);

@mkdir($target_path, 0777, true);

if(move_uploaded_file($file['tmp_name'], $target_file)) {
  printf("%s has been uploaded", $parsedid);
} else{
  echo "there was an error uploading the file, please try again!";
}

?>
