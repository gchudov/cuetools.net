<?php

class phpCTDB{

	private $fp;
  private $fpstats;
  private $atoms;
  public $db;
  public $fulltoc;
  public $mbid;

	function __construct($target_file) {
	  $this->fp = fopen($target_file, 'rb');
		$this->fpstats = fstat($this->fp);
		$this->atoms = $this->parse_container_atom(0, $this->fpstats['size']);
    $this->db = false;
 		foreach ($this->atoms as $entry) if($entry['name'] == 'CTDB') $this->db = $entry;
	}

  function __destruct() {
		fclose($this->fp);
  }

	function ParseTOC()
	{
		$disc = $this->db['discs'][0];
		$this->fulltoc = '';
		$mbtoc = '';
		foreach ($disc['TOC ']['subatoms'] as $track)
		{
			if ($track['name']=='INFO') {
		    $trackcount = phpCTDB::BigEndian2Int(substr($track['value'],0,4));
		    $pregap = phpCTDB::BigEndian2Int(substr($track['value'],4,4));
		    $pos = $pregap + 150;
		  }
		  if ($track['name']=='TRAK') {
		    $isaudio = phpCTDB::BigEndian2Int(substr($track['value'],0,4));
		    $length = phpCTDB::BigEndian2Int(substr($track['value'],4,4));
		    $this->fulltoc = sprintf('%s %d', $this->fulltoc, $pos);
		    $mbtoc = sprintf('%s%08X', $mbtoc, $pos);
		    $pos += $length;
		  }
		}
		$this->fulltoc = sprintf('1 %d %d%s', $trackcount, $pos, $this->fulltoc);
		$mbtoc = sprintf('01%02X%08X%s', $trackcount, $pos, $mbtoc);
		$mbtoc = str_pad($mbtoc,804,'0');
		$this->mbid = str_replace('+', '.', str_replace('/', '_', str_replace('=', '-', base64_encode(pack("H*" , sha1($mbtoc))))));
	}

	static function BigEndian2Int($byte_word, $signed = false) {

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

	static function LittleEndian2String($number, $minbytes=1, $synchsafe=false) {
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

	static function BigEndian2String($number, $minbytes=1, $synchsafe=false) {
	  return strrev(phpCTDB::LittleEndian2String($number, $minbytes, $synchsafe));
	}

  static function discid2path($id)
	{
		$err = sscanf($id, "%03d-%04x%04x-%04x%04x-%04x%04x", $tracks, $id1a, $id1b, $id2a, $id2b, $cddbida, $cddbidb);
		$parsedid = sprintf("%03d-%04x%04x-%04x%04x-%04x%04x", $tracks, $id1a, $id1b, $id2a, $id2b, $cddbida, $cddbidb);
		if ($id != $parsedid)
			die("bad id ". $id);
		return sprintf("parity/%x/%x/%x/%s", $id1b & 15, ($id1b >> 4) & 15, ($id1b >> 8) & 15, $parsedid);
	}

	static function ctdbid2path($discid, $ctdbid)
	{
		$path = phpCTDB::discid2path($discid);
		sscanf($ctdbid, "%04x%04x", $ctdbida, $ctdbidb);
		$parsedctdbid = sprintf("%04x%04x", $ctdbida, $ctdbidb);
		if ($ctdbid != $parsedctdbid)
			die("bad id ". $ctdbid);
		return sprintf("%s/%s.bin", $path, $ctdbid);
	}

	static function unparse_atom($fp, $atom)
	{
//	  printf('unparse_atom(%s)<br>', $atom['name']);
	  $offset = ftell($fp);
	  fwrite($fp, phpCTDB::BigEndian2String(0, 4));
	  fwrite($fp, $atom['name']);
	  if (@$atom['subatoms'])
	    foreach ($atom['subatoms'] as $subatom)
	      phpCTDB::unparse_atom($fp, $subatom);
	  else if ($atom['value'])
	    fwrite($fp, $atom['value']);
	  else
	    die(sprintf("couldn't write long atom %s: size %d", $atom['name'], $atom['size']));
	  $pos = ftell($fp);
	  fseek($fp, $offset, SEEK_SET);
	  fwrite($fp, phpCTDB::BigEndian2String($pos - $offset, 4));
	  fseek($fp, $pos, SEEK_SET);
	}

  function read($offset, $len)
  {
	    fseek($this->fp, $offset, SEEK_SET);
	    return fread($this->fp, $len);
  }

	function parse_container_atom($offset, $len)
	{
//	  printf('parse_container_atom(%d, %d)<br>', $offset, $len);
	  $atoms = false;
	  $fin = $offset + $len;
	  while ($offset < $fin) {
	    fseek($this->fp, $offset, SEEK_SET);
	    $atom_header = fread($this->fp, 8);
	    $atom_size = phpCTDB::BigEndian2Int(substr($atom_header, 0, 4));
	    $atom_name = substr($atom_header, 4, 4);
	    $atom['name'] = $atom_name;
	    $atom['size'] = $atom_size - 8;
	    $atom['offset'] = $offset + 8;
	    if ($atom_size - 8 <= 256)
	      $atom['value'] = fread($this->fp, $atom_size - 8);
	    else
	      $atom['value'] = false;
//    echo $len, ':',  $offset, ":", $atom_size, ":", $atom_name, '<br>';
	    if ($atom_name == 'CTDB' || $atom_name == 'DISC' || $atom_name == 'TOC ' || ($atom_name == 'HEAD' && ($atom_size != 28 || 256 != phpCTDB::BigEndian2Int(substr($atom['value'],0,4)))))
 	   {
	      $atom['subatoms'] = $this->parse_container_atom($offset + 8, $atom_size - 8);
	      foreach ($atom['subatoms'] as $param)
	        switch ($param['name']) {
	          case 'HEAD':
	          case 'TOC ':
	          case 'CRC ':
					  case 'MBID':
					  case 'ART ':
					  case 'nam ':
					  case 'NPAR':
					  case 'CONF':
					  case 'TOTL':
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
	      case 'TOTL':
	        $atom['int'] = phpCTDB::BigEndian2Int($atom['value']);
	        break;
	    }
	    $offset += $atom_size;
	    $atoms[] = $atom;
	  }
	  if ($offset > $fin)
	    die(printf("bad atom: offset=%d, fin=%d", $offset, $fin));
	  return $atoms;
	}
}
?>
