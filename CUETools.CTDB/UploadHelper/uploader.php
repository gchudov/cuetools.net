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

$id = $_POST['id'];
$err = sscanf($id, "%d-%x-%x-%x", $tracks, $id1, $id2, $cddbid);
$target_path = sprintf("parity/%x/%x/%x", $id1 & 15, ($id1 >> 4) & 15, ($id1 >> 8) & 15);
$target_file = sprintf("%s/dBCT-%03d-%08x-%08x-%08x.bin", $target_path, $tracks, $id1, $id2, $cddbid);

@mkdir($target_path, 0777, true);

if(move_uploaded_file($file['tmp_name'], $target_file)) {
    echo "The file ".  $target_file. " has been uploaded";
} else{
    echo "There was an error uploading the file, please try again!";
}

?>
