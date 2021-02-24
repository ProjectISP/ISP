// Publish project specific data
(function() {
rh = window.rh;
model = rh.model;
var defaultTopic = "Home/Introduction.htm";
rh._.exports(defaultTopic);
rh.consts('DEFAULT_TOPIC', encodeURI("Home/Introduction.htm"));
rh.consts('HOME_FILEPATH', encodeURI('index.htm'));
rh.consts('START_FILEPATH', encodeURI('index.htm'));
rh.consts('HELP_ID', '0f627a1a-9237-40f2-8414-c448d725d75a' || 'preview');
rh.consts('LNG_SUBSTR_SEARCH', 0);

model.publish(rh.consts('KEY_LNG_NAME'), "en");
model.publish(rh.consts('KEY_DIR'), "ltr");
model.publish(rh.consts('KEY_LNG'), {"Contents":"Contents","Index":"Index","Search":"Search","Glossary":"Glossary","Logo/Author":"Powered By","Show":"Show","Hide":"Hide","SyncToc":"SyncToc","Prev":"Previous","Next":"Next","Disabled Prev":"<<","Disabled Next":">>","Separator":"|","OpenLinkInNewTab":"Open in new tab","SearchOptions":"Search Options","Loading":"Loading...","UnknownError":"Unknown error","Logo":"Logo","HomeButton":"Home","SearchPageTitle":"Search Results","PreviousLabel":"Previous","NextLabel":"Next","TopicsNotFound":"No results found","JS_alert_LoadXmlFailed":"Failed to load XML file","JS_alert_InitDatabaseFailed":"Failed to initialize database","JS_alert_InvalidExpression_1":"The search string you typed is not valid.","Searching":"Searching...","Cancel":"Cancel","Canceled":"Canceled","ResultsFoundText":"%1 result(s) found for %2","SearchResultsPerScreen":"Search results per page","Back":"Back","TableOfContents":"Table of Contents","IndexFilterKewords":"Filter Keywords","GlossaryFilterTerms":"Filter Terms","ShowAll":"All","HideAll":"Hide All","ShowHide":"Show/Hide","IeCompatibilityErrorMsg":"This page cannot be viewed in Internet Explorer 8 or earlier version.","NoScriptErrorMsg":"Enable JavaScript support in the browser to view this page.","EnableAndSearch":"Include all words in search","HighlightSearchResults":"Highlight search results","Print":"Print","Filter":"Filter","SearchTitle":"Search","ContentFilterChanged":"Content filter is changed, search again","EndOfResults":"End of search results.","Reset":"Reset","NavTip":"Close","ToTopTip":"Go to top","ApplyTip":"Apply","SidebarToggleTip":"Expand/Collapse","Copyright":"© Copyright 2019. All rights reserved.","FavoriteBoxTitle":"Favorites","setAsFavorites":"Add to Favorites","unsetAsFavorite":"Unset as favorite","favoritesNameLabel":"Name","favoritesLabel":"Favorites","setAsFavorite":"Set as Favorite","nofavoritesFound":"You have not marked any topic as favorite.","Welcome_header":"Welcome to our Help Center","Welcome_text":"What can we help you with today?","SearchButtonTitle":"Search for...","ShowTopicInContext":"Click here to see this page in full context","TopicHiddenText":"This topic is filtered out by the selected filters.","NoTermsFound":"No terms found","NoKeywordFound":"No keyword found","SkipToMainContent":"Skip To Main Content"});

model.publish(rh.consts('KEY_HEADER_TITLE'), "Integrated Seismic Programs");
model.publish(rh.consts('PDF_FILE_NAME'), "");
model.publish(rh.consts('MAX_SEARCH_RESULTS'), "20");
model.publish(rh.consts('KEY_SKIN_FOLDER_NAME'), "Red_Green");
model.publish(rh.consts('CHAT_API_SESSION_TOKEN'), "");
model.publish(rh.consts('CHAT_API_PROJ_ID'), "");

model.publish(rh.consts('KEY_SUBSTR_SEARCH'), "true");
})();
