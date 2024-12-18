# Changelog

## 2.4.0 (2024-12-18)


### Features

* Add base support for audio sources ([0f24bd2](https://github.com/OzGav/server/commit/0f24bd2a58e0242e8415d125adf733254b944ad1))
* Adjust code to changes in the models ([b1a67a4](https://github.com/OzGav/server/commit/b1a67a46ae3b20b1cfa707bdbe9ffe0dd3d52bba))
* Adjust code to changes in the models ([268ffb5](https://github.com/OzGav/server/commit/268ffb51abcc6afd90fad2ceaa83fd19687b01c4))
* Bump models to 1.1.2 ([b1a67a4](https://github.com/OzGav/server/commit/b1a67a46ae3b20b1cfa707bdbe9ffe0dd3d52bba))
* Bump models to 1.1.2 ([b1a67a4](https://github.com/OzGav/server/commit/b1a67a46ae3b20b1cfa707bdbe9ffe0dd3d52bba))
* Bump models to 1.1.2 ([268ffb5](https://github.com/OzGav/server/commit/268ffb51abcc6afd90fad2ceaa83fd19687b01c4))
* Bump models to 1.1.2 ([268ffb5](https://github.com/OzGav/server/commit/268ffb51abcc6afd90fad2ceaa83fd19687b01c4))
* Bump models to 1.1.3 ([fe175c4](https://github.com/OzGav/server/commit/fe175c4fae9ea431c47c86c1841eee62f1d4b2fe))
* pre-install all requirements in the docker image ([7c69c66](https://github.com/OzGav/server/commit/7c69c66af9ac488d5a2882cab1278bd662ba2ddf))
* Radio mode enhancements ([#1654](https://github.com/OzGav/server/issues/1654)) ([a0abddc](https://github.com/OzGav/server/commit/a0abddc6a60f409e14dc5fa317a3522be14a8ec3))
* Refactor sync to group to make it more universal ([b1a67a4](https://github.com/OzGav/server/commit/b1a67a46ae3b20b1cfa707bdbe9ffe0dd3d52bba))
* Refactor sync to group to make it more universal ([268ffb5](https://github.com/OzGav/server/commit/268ffb51abcc6afd90fad2ceaa83fd19687b01c4))
* Reorganize repository to contain only the server code ([36a706f](https://github.com/OzGav/server/commit/36a706fcdf7a5a16e04fef5fe0d599b1ef1d37d3))
* Use dedicated directory for cache files ([5ab2d36](https://github.com/OzGav/server/commit/5ab2d369f1a606b80a2c170df5c3218281c425cf))


### Bug Fixes

* 🐛 explicitly add dist to build context, ls contents ([#1648](https://github.com/OzGav/server/issues/1648)) ([afc29d9](https://github.com/OzGav/server/commit/afc29d93b6632857d60b0cbeaa9ece49b2905199))
* 🐛 Hopefully fix release ([#1644](https://github.com/OzGav/server/issues/1644)) ([5e1ca3c](https://github.com/OzGav/server/commit/5e1ca3cc21c2413f5ae5d0f83f0fcd7f1c05b577))
* 🐛 move download to after checkout, fix conditionals ([#1645](https://github.com/OzGav/server/issues/1645)) ([930f2d7](https://github.com/OzGav/server/commit/930f2d799e001259b63be29a4ebaf0fc10f09a46))
* account for playlists with less than 50 images/items ([02a5dfc](https://github.com/OzGav/server/commit/02a5dfce0b52599d998f77c77fb0bbd6dd9476b1))
* Add logging around Subsonic scrobble calls ([#1797](https://github.com/OzGav/server/issues/1797)) ([c33e766](https://github.com/OzGav/server/commit/c33e766ec4062ac9b812d31968fe3393bada4df6))
* album directory parsing with album versions ([#1683](https://github.com/OzGav/server/issues/1683)) ([ec554ec](https://github.com/OzGav/server/commit/ec554ecf127b9df47ee875a1fde61b6a29ff577d))
* announcements on HA players ([ec9c476](https://github.com/OzGav/server/commit/ec9c4766037e47852a9a1b0141d41ad8eb1f8a1b))
* Can't delete item in queue when queue is not active ([07df754](https://github.com/OzGav/server/commit/07df754321468a3192b80bfcafa4f0252745eab1))
* Check if metadata job exists before clearing it ([0b78ba0](https://github.com/OzGav/server/commit/0b78ba05c9f7bc8e1b03ce28000c4d5e4cc44080))
* Correct maxsize of `MetadataLookupQueue` ([#1805](https://github.com/OzGav/server/issues/1805)) ([cf77b99](https://github.com/OzGav/server/commit/cf77b99f3fabe559a97c891595bc6c292dbbb8fe))
* Disable shuffle for radio mode ([#1673](https://github.com/OzGav/server/issues/1673)) ([191200b](https://github.com/OzGav/server/commit/191200bd0e141217da397f25248d86f5e35f06c0))
* Do not retry Snapcast connection if we want to exit ([6ad7503](https://github.com/OzGav/server/commit/6ad750347962d73a281a2cef4f88f95627aab32f))
* don't enqueue next if flow mode is enabled ([12e2ccc](https://github.com/OzGav/server/commit/12e2ccc81eb05644230218d67107cf64b747d1a9))
* Don't Stop the Music for Subsonic and remove Podcast hacks ([#1774](https://github.com/OzGav/server/issues/1774)) ([da42973](https://github.com/OzGav/server/commit/da429731696e68770c2992837029810e58deeb6d))
* Enqueue player feature not correctly set on cast groups and dlna players ([f9a855d](https://github.com/OzGav/server/commit/f9a855de1f846de5c582c346d0cfbaa8394ba1e8))
* ensure Spotify token is fresh when retrieving streamdetails ([48a6983](https://github.com/OzGav/server/commit/48a698316a78d00ee686c7f05d483e10bee221f5))
* faster retry on spotify token expiration ([c1aadac](https://github.com/OzGav/server/commit/c1aadac77b8b53a0ccdde0131ef5619402051884))
* flow mode not being applied in all cases (while it should) ([#1672](https://github.com/OzGav/server/issues/1672)) ([259252f](https://github.com/OzGav/server/commit/259252f244a9bd1b4f8bbc4c74d73ee41ec51668))
* Group volume up/down not implemented ([a1eede9](https://github.com/OzGav/server/commit/a1eede9b1675a52bb8cc3e03a95a449754e2255b))
* Handle radio stations providing non utf-8 in streamtitle ([#1664](https://github.com/OzGav/server/issues/1664)) ([605b09f](https://github.com/OzGav/server/commit/605b09fa381827b9d6a348237a7b7d0e5e9dadeb))
* Handle retry exception during authentication in Bluesound provider ([#1778](https://github.com/OzGav/server/issues/1778)) ([4744d18](https://github.com/OzGav/server/commit/4744d18d7325097d9ef51a7955f987b4fed042f2))
* Handle some small race conditions with sonos players ([d150b87](https://github.com/OzGav/server/commit/d150b8750ffa3c8c63c4396abc21eae1840fdf92))
* Ignore airplay for broken Sonos devices ([0d76599](https://github.com/OzGav/server/commit/0d76599d1608c4c70274d9bd9ef293b8d0de5d04))
* Improve accuracy of matching album in directory structure for local filesystem ([#1779](https://github.com/OzGav/server/issues/1779)) ([324d360](https://github.com/OzGav/server/commit/324d360f53eafac7dfb793053734e8abc2596990))
* Issues with player groups and airplay mode ([790c6aa](https://github.com/OzGav/server/commit/790c6aaf0e5fdd044a91ff1c9e5f72dc4d5b2717))
* leftover issues after repo split up ([cec5e7f](https://github.com/OzGav/server/commit/cec5e7f7c55fbb18185c0e72c93492b00b574bf1))
* playerqueue elapsed time not changing for cast players ([9fddafd](https://github.com/OzGav/server/commit/9fddafd874d80917e3b553f4787a3d2562f7287e))
* Prevent playlist collage image take up all system memory ([f38770e](https://github.com/OzGav/server/commit/f38770efc429911b2a94f1ef1b65586c3c2502a2))
* Prevent redundant lookup of full media item in queue controller ([734dc5b](https://github.com/OzGav/server/commit/734dc5b99b705ed9ce6089d85ee72f4484836442))
* Radio mode for Subsonic provider ([#1784](https://github.com/OzGav/server/issues/1784)) ([c028d5c](https://github.com/OzGav/server/commit/c028d5cb3ab385580120003b1d7bc7b832fd53e3))
* Reauthenticate with SiriusXM when playing a station ([#1789](https://github.com/OzGav/server/issues/1789)) ([32667fb](https://github.com/OzGav/server/commit/32667fbc8fd56d21d0699d5267f6fe49f94064ca))
* snapcast player in universal player group ([#1756](https://github.com/OzGav/server/issues/1756)) ([f7ee100](https://github.com/OzGav/server/commit/f7ee1007d79ab681c014e540da69d91e6457f4f1))
* Solved a bug for applying genres in Soundcloud tracks ([#1803](https://github.com/OzGav/server/issues/1803)) ([cef1536](https://github.com/OzGav/server/commit/cef153656735d866ee5ea1654a4d52d6cdaf63cc))
* some small typos and optimizations ([02ee5a3](https://github.com/OzGav/server/commit/02ee5a3f1763e95c944f3ec74f6bad17a3553e46))
* Sonos Airplay mode ([2a90217](https://github.com/OzGav/server/commit/2a9021764ef830fb0bbebc25f3d79a7af5959ae9))
* Sonos airplay mode infinite loop ([c710d9b](https://github.com/OzGav/server/commit/c710d9bced22f38fb33663f16b6b24e3cb1567d6))
* Subsonic: Allow user to force player provider seek ([#1798](https://github.com/OzGav/server/issues/1798)) ([10e1378](https://github.com/OzGav/server/commit/10e13786601d84c6d1e89b458899861f0e8f1296))
* Tweaks for ESPHome mediaplayers ([46fe6ab](https://github.com/OzGav/server/commit/46fe6ab9f58e6ab84c9c77e7edb8323fb3b383a4))
* use relative path for all default images and collages ([fdc942f](https://github.com/OzGav/server/commit/fdc942fa3e853f5cd9c47ca42b3c1fe28ee623e6))


### Miscellaneous Chores

* release 2.4.0b1 ([1b6b216](https://github.com/OzGav/server/commit/1b6b216a9d1c6f1be4909d7876361c1dac3bc16e))
* release 2.4.0b4 ([f0ff2f6](https://github.com/OzGav/server/commit/f0ff2f62c3b50753c9e9817c5051703637f3f18e))
* release 2.4.0b6 ([16c2479](https://github.com/OzGav/server/commit/16c24799a564f86074227bf07fde14e21384658e))
