use proc_macro::TokenStream;
use proc_macro_crate::{
    FoundCrate,
    crate_name,
};
use proc_macro2::Span;
use quote::{
    format_ident,
    quote,
};
use syn::{
    Data,
    DeriveInput,
    Fields,
    Ident,
    parse_macro_input,
};

#[proc_macro_derive(Parameters, attributes(parameter))]
pub fn derive_parameters(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    match expand_parameters(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn expand_parameters(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let cc_path = cc_problems_path();
    let name = &input.ident;
    let ids_name = format_ident!("{name}Ids");

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    &data.fields,
                    "Parameters requires named struct fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(input, "Parameters can only be derived for structs"));
        }
    };

    let offset = Ident::new("offset", Span::call_site());
    let mut id = quote!(0usize);
    let mut id_fields = Vec::new();
    let mut id_values = Vec::new();
    let mut registry_values = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        if is_nested(field) {
            id_fields.push(quote! {
                pub #ident: <#ty as #cc_path::parameters::Parameters>::Ids
            });

            id_values.push(quote! {
                #ident: <#ty as #cc_path::parameters::Parameters>::ids_at(#offset + #id)
            });

            registry_values.push(quote! {
                registry.extend(self.#ident.registry());
            });
            id = quote!(#id + <#ty as #cc_path::parameters::Parameters>::PARAM_COUNT);
        } else {
            id_fields.push(quote! {
                pub #ident: #cc_path::parameters::TypedParamId<#ty>
            });

            id_values.push(quote! {
                #ident: #cc_path::parameters::TypedParamId::new(#offset + #id)
            });

            registry_values.push(quote! {
                registry.push(self.#ident.clone());
            });
            id = quote!(#id + 1);
        }
    }

    Ok(quote! {
        pub struct #ids_name {
            #(#id_fields,)*
        }

        impl #cc_path::parameters::Parameters for #name {
            type Ids = #ids_name;

            fn ids_at(offset: usize) -> Self::Ids {
                #ids_name {
                    #(#id_values,)*
                }
            }

            const PARAM_COUNT: usize = #id;

            fn registry(&self) -> #cc_path::parameters::ParameterRegistry {
                let mut registry = #cc_path::parameters::ParameterRegistry::default();

                #(#registry_values)*

                registry
            }
        }
    })
}

fn is_nested(field: &syn::Field) -> bool {
    field.attrs.iter().any(|attr| {
        if !attr.path().is_ident("parameter") {
            return false;
        }

        let mut nested = false;

        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("nested") {
                nested = true;
            }

            Ok(())
        });

        nested
    })
}

fn cc_problems_path() -> proc_macro2::TokenStream {
    match crate_name("cc_problems") {
        Ok(FoundCrate::Itself) => quote!(crate),
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name, proc_macro2::Span::call_site());
            quote!(::#ident)
        }
        Err(_) => {
            panic!("could not find `cc_problems` crate");
        }
    }
}
