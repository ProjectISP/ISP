//Variables to override in settings

var useTOC = true;
var useGLO = true;
var useIDX = true;
var useFilter = true;
var useFacebook = true;
var useTwitter = true;
var useANDsearch = true;
var fontFamily = "\"Trebuchet MS\", Arial, sans-serif";

var delayLoadIdx = true;
var delayLoadGlo = true;
var useSocial = true;
var maxResults = 15;
var titleColor = "#ffffff";
var backgroundColor = "#509de6";

(function() {
  var rh = window.rh, model = rh.model

  model.publish(rh.consts('KEY_DIR'), "ltr");
  model.publish(rh.consts('KEY_IS_RESPONSIVE'), true);
  model.publish(rh.consts("KEY_MOBILE_TOC_DRILL_DOWN"), true);
  model.publish(rh.consts('KEY_DEFAULT_SEARCH_LOCATION'), "tabbar");
  model.publish(rh.consts('KEY_SEARCH_HIGHLIGHT_COLOR'), "#000000");
  model.publish(rh.consts('KEY_SEARCH_BG_COLOR'), "#FCFF00");
	model.publish('l.desktop_sidebar_visible', true);	
	model.publish('l.mobile_header_visible', false);
	model.publish(rh.consts('KEY_DO_NOT_PRESERVE_AR'), true);
	model.publish(rh.consts('KEY_CUSTOM_BUTTONS_CONFIG'), [{"image":"expand_all.svg","name":"Expand/Collapse All","onclick":"rh.model.publish(rh.consts('EVT_EXPAND_COLLAPSE_ALL'));return false;","title":"Expand/Collapse All"},{"image":"removesearch_mark.svg","name":"RemoveHighlight","onclick":"rh.model.publish(rh.consts('EVT_REMOVE_HIGHLIGHT'));return false;","title":"Remove Highlight"},{"image":"print_desktop.png","name":"Print","onclick":"rh.model.publish(rh.consts('EVT_PRINT_TOPIC'));return false;","title":"Print"}])

  model.subscribe([rh.consts('KEY_DEFAULT_SEARCH_LOCATION'), rh.consts('KEY_FEATURE')], function() {
    var features = model.get(rh.consts('KEY_FEATURE')) || {},
      searchResultInTabbar = model.get(rh.consts('KEY_DEFAULT_SEARCH_LOCATION')) === 'tabbar'
    if (!features.toc && !features.idx && !features.glo && !features.filter && !searchResultInTabbar) {
      model.publish('l.desktop_sidebar_available', false);
    } else {
      model.publish('l.desktop_sidebar_available', true);
    }
  });

  var phone_max_width = 941;
  var tablet_max_width = 1295;
  var screens = {
	  ios: {user_agent: /(iPad|iPhone|iPod)/g}
	};

  screens.phone =  { media_query: 'screen and (max-width: '+ phone_max_width +'px)' };
  if(phone_max_width === 0) {
    screens.tablet =  { media_query: 'screen and (max-width: '+ tablet_max_width +'px)' };
  } else {
    screens.tablet =  { media_query: 'screen and (min-width: '+ (phone_max_width + 1) +'px) and (max-width: '+ tablet_max_width +'px)' };
  }
  if(tablet_max_width === 0) {
    screens.desktop =  { media_query: 'screen and (min-width: '+ (phone_max_width || 1) +'px)' };
  } else {
    screens.desktop =  { media_query: 'screen and (min-width: '+ (tablet_max_width + 1) +'px)' };
  }
  model.publish(rh.consts('KEY_SCREEN'), screens);
  model.publish(rh.consts('KEY_DEFAULT_SCREEN'), "phone");
}.call(this));

(function() {
	var mobileMenu, rh, features;

	rh = window.rh;
	features = rh.model.get(rh.consts('KEY_FEATURE')) || {};

	//Publish which panes are available
	features.toc = useTOC;
	features.idx = useIDX;
	features.glo = useGLO;
	features.filter = useFilter;

  features.delay_load_idx = delayLoadIdx;
  features.delay_load_glo = delayLoadGlo;

	rh.model.publish(rh.consts('KEY_DEFAULT_TAB'), "toc");

	//If there are are no panes available
	if (!useTOC && !useGLO && !useIDX) {
		mobileMenu = false;
	} else {
		mobileMenu = true;
	}

	rh.model.publish('l.mobile_menu_enabled', mobileMenu);

	//Number of search results to be loaded at once.
	rh.consts('MAX_RESULTS', '.l.maxResults');
	rh.model.publish(rh.consts('MAX_RESULTS'), maxResults);

	//Choose whether to use the AND search option in the layout
	features.andsearch = useANDsearch;

	/* This layout has single page and so handles search */
	rh.model.publish(rh.consts("KEY_CAN_HANDLE_SEARCH"), true);

	//Social widgets
	if(document.location.toString().indexOf("file:///") != -1) {//Always disable buttons for local output
		useFacebook = false;
		useTwitter = false;
		useSocial = false;
	}
	if(!useFacebook && !useTwitter) {
		useSocial = false;
	}

	features.facebook = useFacebook;
	features.twitter = useTwitter;
	features.social = useSocial;

	if(useFacebook) {//Facebook Button
		rh.model.subscribe(rh.consts('KEY_TOPIC_TITLE'), updateFacebookButton);
		rh.model.subscribe('l.social_opened', updateFacebookButton);
	}
	if(useTwitter) {//Twitter button
		rh.model.subscribe(rh.consts('KEY_TOPIC_TITLE'), updateTwitterButton);
		rh.model.subscribe('l.social_opened', updateTwitterButton);
	}
	function updateFacebookButton() {
		var iframeID, url, iframe, topicUrl;

		topicUrl = rh.model.get(rh.consts('KEY_TOPIC_URL'));

		if(!rh.model.get('l.social_opened') || !topicUrl) {
			return;
		}

		if(document.location.toString().indexOf("file://") != -1) {
			return;//No FB button on local content
		}

		iframeID = "bf-iframe";

		//The URL for the Facebook iFrame
		url = 'http://www.facebook.com/plugins/share_button.php?href='+
			  topicUrl +
			  '&layout=button_count&action=like&show_faces=false&share=false&height=21';

		iframe = document.getElementById(iframeID);
		iframe.setAttribute("src", url);

	}
	function updateTwitterButton() {
		var holderID, holder, newLink, textNode, topicUrl;

		topicUrl = rh.model.get(rh.consts('KEY_TOPIC_URL'));

		if(!rh.model.get('l.social_opened') || !topicUrl) {
			return;
		}

		if(document.location.toString().indexOf("file://") != -1) {
			return;//No Tweet button on local content
		}

		holderID = 'twitter-holder';
		holder = document.getElementById(holderID);

		//Remove existing children
		while (holder.firstChild) {
			holder.removeChild(holder.firstChild);
		}

		//Add tweet button
		newLink = document.createElement('a');
		newLink.setAttribute("href", 'https://twitter.com/share');
		newLink.setAttribute("class", 'twitter-share-button');
		newLink.setAttribute("data-url", topicUrl);
		newLink.setAttribute("data-text", rh.model.get(rh.consts('KEY_TOPIC_TITLE')));

		textNode = document.createTextNode("Tweet");
		newLink.appendChild(textNode);

		holder.appendChild(newLink);

		if(window.twttr) {
			window.twttr.widgets.load();
		}
	}

	rh.model.publish(rh.consts('KEY_FEATURE'), features);
	rh.model.publish(rh.consts("KEY_LAYOUT_VERSION"), "3.0");
}.call(this));
