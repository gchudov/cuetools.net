<?php
$target_path = "parity/";
$target_path = $target_path . basename( $_FILES['uploadedfile']['name']); 
$source_path = $_FILES['uploadedfile']['tmp_name'];

if(move_uploaded_file($_FILES['uploadedfile']['tmp_name'], $target_path)) {
    echo "The file ".  basename( $_FILES['uploadedfile']['name']). 
    " has been uploaded";
} else{
    echo $target_path, ":", $source_path;
    echo "There was an error uploading the file, please try again!";
}

?>
