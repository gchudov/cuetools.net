<?php include 'logo_start.php'; ?>
<h2>What's it for?</h2>
You probably heard about <a href=http://www.accuraterip.com/>AccurateRip</a>, a wonderfull database of CD rip checksums, which helps you make sure your CD rip is an exact copy of original CD. What it can tell you is how many other people got the same data when copying this CD.

CUETools Database is an extension of this idea.

<h2>What are the advantages?</h2>
<ul>
<li>The most important feature is the ability not only to detect, but also correct small amounts of errors that occured in the ripping process.
<li>It's free of the offset problems. You don't even need to set up offset correction for your CD drive to be able to verify and what's more important, submit rips to the database. Different pressings of the same CD are treated as the same disc by the database, it doesn't care.
<li>Verification results are easier to deal with. There are exactly three possible outcomes: rip is correct, rip contains correctable errors, rip is unknown (or contains errors beyond repair).
<li>If there's a match, you can be certain it's really a match, because in addition to recovery record database uses a well-known CRC32 checksum of the whole CD image (except for 10*588 offset samples in the first and last seconds of the disc). This checksum is used as a rip ID in CTDB.
</ul>
<h2>What are the downsides and limitations?</h2>
<ul>
<li>CUETools DB doesn't bother with tracks. Your rip as a whole is either good/correctable, or it isn't. If one of the tracks is damaged beyound repair, CTDB cannot tell which one.
<li>If your rip contains errors, verification/correction process will involve downloading about 200kb of data, which is much more than it takes for AccurateRp.
<li>Verification process is slower than with AR.
<li>Database was just born and at the moment contains much less CDs than AR.
</ul>
<h2>How many errors can a rip contain and still be repairable?</h2>
<ul>
<li>That depends. The best case scenario is when there's one continuous damaged area up to 30-40 sectors (about half a second) long.
<li>The worst case scenario is 4 non-continuous damaged sectors in (very) unlucky positions.
</ul>
<h2>What information does the database contain per each submission?</h2>
<ul>
<li>CD TOC (Table Of Contents), i.e. length of every track.
<li>Offset-finding checksum, i.e. small (16 byte) recovery record for a set of samples throughout the CD, which allows to detect the offset difference between the rip in database and your rip, even if your rip contains some errors.
<li>CRC32 of the whole disc (except for some leadin/leadout samples).
<li>Submission date, artist, title.
<li>180kb recovery record, which is stored separately and accessed only when verifying a broken rip or repairing it.
</ul>
</body>
</html>
