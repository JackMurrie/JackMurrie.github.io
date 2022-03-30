# coding: utf-8

Gem::Specification.new do |spec|
  spec.name                    = "jekyll-blog"
  spec.version                 = "1.0.0"
  spec.authors                 = ["Jack Murrie"]

  spec.summary                 = %q{A flexible two-column Jekyll theme.}
  spec.homepage                = "https://jackmurrie.github.io//"
  spec.license                 = "MIT"

  spec.metadata["plugin_type"] = "theme"

  spec.files                   = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^(assets|_(data|includes|layouts|sass)/|(LICENSE|README|CHANGELOG)((\.(txt|md|markdown)|$)))}i)
  end

  spec.add_runtime_dependency "jekyll", "~> 3.6"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-sitemap", "~> 1.1"
  spec.add_runtime_dependency "jekyll-gist", "~> 1.4"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.9.2"
  spec.add_runtime_dependency "jekyll-data", "~> 1.0"
  spec.add_runtime_dependency "jemoji", "~> 0.8"

  spec.add_development_dependency "bundler", "~> 2.3.10"
  spec.add_development_dependency "rake", "~> 12.3.3"
end
